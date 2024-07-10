#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy
import os
from random import randint
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import get_max_alphas
from gaussian_renderer import render as gaussian_render
# from gaussian_renderer.compression_renderer import render

from gaussian_renderer.covariance_renderer import render as render
from models.compression.compression_utils import get_size_in_bits
from models.compression.multirate_complete_eb import CompleteEntropyBottleneck
from models.splatting.hierarchical.hierarchy_utils import choose_min_max_depth
from models.splatting.mcmc_model import GaussianModel
from scene import Scene
from scene.cameras import Camera
from training.training_utils import bump_iterations_for_secondary_training
from utils.general_utils import decompose_covariance_matrix, make_psd, compute_required_rotation_and_scaling, apply_cholesky, apply_inverse_cholesky, pack_full_to_lower_triangular, unpack_lower_triangular_to_full, apply_rotation_and_scaling
from utils.loss_utils import l1_loss, l2_loss, ssim


def freeze_geometry(gaussians, dataset):
    if dataset.freeze_geometry:
        gaussians._xyz.requires_grad_(False)
        gaussians._scaling.requires_grad_(False)
        gaussians._rotation.requires_grad_(False)
        gaussians._opacity.requires_grad_(False)

        if hasattr(gaussians, "_mask"):
            gaussians._mask.requires_grad_(False)
    else:
        gaussians._xyz.requires_grad_(True)
        gaussians._scaling.requires_grad_(True)
        gaussians._rotation.requires_grad_(True)
        gaussians._opacity.requires_grad_(True)

        if hasattr(gaussians, "_mask"):
            gaussians._mask.requires_grad_(True)

    gaussians._features_dc.requires_grad_(True)
    gaussians._features_rest.requires_grad_(True)


def training(
    dataset,
    optimization,
    pipeline,
    compressor: CompleteEntropyBottleneck,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    logger,
    debug_from,
):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(optimization)

    if compressor is not None:
        compressor.training_setup(optimization)

    if checkpoint:
        loaded_checkpoint = torch.load(checkpoint)

        if len(loaded_checkpoint) == 2:
            (model_params, first_iter) = loaded_checkpoint
        elif len(loaded_checkpoint) == 3:
            (model_params, compressor_params, first_iter) = loaded_checkpoint
            compressor.restore(compressor_params)
        else:
            raise ValueError("Invalid checkpoint format")

        gaussians.restore(model_params, optimization)
        compressor.intra_compressor.init_normalization_params(gaussians)
        compressor.inter_compressor.init_normalization_params(gaussians)

        if dataset.freeze_geometry:
            freeze_geometry(gaussians, dataset)

    bump_iterations_for_secondary_training(optimization, first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    COMPRESSION_TRAIN_ITER_FROM = dataset.compression_training_from
    entropy_lambdas = dataset.lambdas_entropy
    subsample_fraction = dataset.compression_subsample_fraction
    # subsample_fraction = 0.7
    print(f"Entropy lambdas: {entropy_lambdas}")

    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_loss_for_pos = 0.0
    ema_loss_for_covariance = 0.0
    ema_loss_for_sh = 0.0
    ema_loss_for_opacity = 0.0

    progress_bar = tqdm(range(first_iter, optimization.iterations), desc="Training progress")
    first_iter += 1

    with torch.no_grad():
        # Calculate the hierarchy just once, never update Gaussians
        chosen_depth, max_depth = choose_min_max_depth(gaussians.get_xyz)

        (mean3D, covariance, shs, opacity, node_assignment) = compressor.get_cut_attributes(
            gaussians, chosen_depth, max_depth
        )
        covariance_L = apply_cholesky(covariance)

        # scaling0, rotation0 = decompose_covariance_matrix(covariance[: covariance.shape[0] // 2])
        # scaling1, rotation1 = decompose_covariance_matrix(covariance[covariance.shape[0] // 2 :])
        # scaling = torch.cat((scaling0, scaling1), dim=0)
        # rotation = torch.cat((rotation0, rotation1), dim=0)

        # Sort the node assignments for more efficient assignment compression
        sorted_node_assignments, sorted_indices = torch.sort(node_assignment)
        # compute_required_rotation_and_scaling(covariance[node_assignment], gaussians.get_covariance())

        np.savez_compressed(
            f"{dataset.model_path}/node_assignments.npz",
            node_assignment=sorted_node_assignments.cpu().numpy(),
        )
        # np.savez_compressed(
        #     f"{dataset.model_path}/aggregated_means3D.npz",
        #     mean3D=mean3D.cpu().numpy(),
        # )
        logger.info(f"Number of Gaussians: {gaussians._xyz.shape[0]}")
        logger.info(f"Number of aggregated Gaussians: {mean3D.shape[0]}")
        node_assignment_size = os.path.getsize(f"{dataset.model_path}/node_assignments.npz")
        logger.info(f"Node assignments size: {node_assignment_size / (2**20):.{5}f} MB")
        # agg_means_size = os.path.getsize(f"{dataset.model_path}/aggregated_means3D.npz")
        # logger.info(f"Aggregated means3D size: {agg_means_size / (2**20):.{5}f} MB")

        # covariance_transform_parameters = compute_required_rotation_and_scaling(
        #     covariance[node_assignment], gaussians.get_covariance()
        # )

    for iteration in range(first_iter, optimization.iterations + 1):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipeline.debug = True

        bg = torch.rand((3), device="cuda") if optimization.random_background else background

        entropy_lambda_level = np.random.randint(0, len(entropy_lambdas))
        # trial_covariance = torch.zeros((res_covariance.shape[0], 6), device="cuda")

        ########################################################################
        # 1. Perform intra-coding of the aggregated attributes
        intra_indices = compressor.get_subsample_indices(subsample_fraction, mean3D.shape[0])

        # breakpoint()

        intra_input = torch.cat(
            (
                covariance_L[intra_indices],
                opacity[intra_indices],
                shs[intra_indices].view(-1, 3 * ((dataset.sh_degree + 1) ** 2)),
            ),
            dim=1,
        )
        intra_compressed_attrs, intra_likelihoods, intra_num_coeffs = (
            compressor.forward_intra_params(intra_input, level=entropy_lambda_level)
        )
        intra_num_coeffs = len(intra_indices) * intra_num_coeffs

        intra_covariance = apply_inverse_cholesky(intra_compressed_attrs[:, :6])

        compressed_agg_mean3D = mean3D.clone()
        compressed_agg_covariance = covariance.clone()
        compressed_agg_opacity = opacity.clone()
        compressed_agg_shs = shs.clone()

        compressed_agg_covariance[intra_indices] = make_psd(intra_covariance)
        compressed_agg_opacity[intra_indices] = torch.clamp(intra_compressed_attrs[:, 6:7], 0, 1)
        compressed_agg_shs[intra_indices] = intra_compressed_attrs[:, 7:].view(
            intra_compressed_attrs.shape[0], -1, 3
        )
        ########################################################################

        ########################################################################
        # 2. Calculate residuals for the compressed attributes
        inter_indices = compressor.get_subsample_indices(subsample_fraction, mean3D.shape[0])

        res_position, res_scaling, res_rotation, res_shs, res_opacity = compressor.get_residuals(
        # res_position, res_covariance, res_shs, res_opacity = compressor.get_residuals(
            gaussians.get_xyz[sorted_indices],
            gaussians.get_covariance()[sorted_indices],
            gaussians.get_features[sorted_indices],
            gaussians.get_opacity[sorted_indices],
            compressed_agg_mean3D[sorted_node_assignments].detach(),
            compressed_agg_covariance[sorted_node_assignments].detach(),
            compressed_agg_shs[sorted_node_assignments].detach(),
            compressed_agg_opacity[sorted_node_assignments].detach()
        )
        ########################################################################

        ########################################################################
        # 3. Perform inter-coding of the residuals
        inter_input = torch.cat(
            (
                res_position[inter_indices],
                # res_covariance[inter_indices],
                res_scaling[inter_indices],
                res_rotation[inter_indices],
                res_opacity[inter_indices],
                res_shs[inter_indices].view(-1, 3 * ((dataset.sh_degree + 1) ** 2)),
            ),
            dim=1,
        )
        inter_compressed_attrs, inter_likelihoods, res_num_coeffs = compressor.forward_inter_params(
            inter_input, level=entropy_lambda_level
        )
        res_num_coeffs = len(inter_indices) * res_num_coeffs

        out_res_pos = res_position.clone()
        out_res_shs = res_shs.clone()
        out_res_opacity = res_opacity.clone()
        # out_res_covariance = res_covariance.clone()

        out_res_pos[inter_indices] = inter_compressed_attrs[:, :3]
        # out_res_covariance[inter_indices] = inter_compressed_attrs[:, 3:9]
        out_res_opacity[inter_indices] = inter_compressed_attrs[:, 10:11]
        out_res_shs[inter_indices] = inter_compressed_attrs[:, 11:].view(
            inter_compressed_attrs.shape[0], -1, 3
        )

        full_covariance = gaussians.get_covariance()[sorted_indices]
        full_covariance[inter_indices] = make_psd(
            apply_rotation_and_scaling(
                compressed_agg_covariance[sorted_node_assignments][inter_indices].detach(),
                torch.abs(inter_compressed_attrs[:, 3:6]),
                torch.nn.functional.normalize(inter_compressed_attrs[:, 6:10]),
            )
        )
        full_position = compressed_agg_mean3D[sorted_node_assignments].detach() + out_res_pos
        # full_covariance = make_psd(compressed_agg_covariance[sorted_node_assignments].detach() + out_res_covariance)
        full_shs = compressed_agg_shs[sorted_node_assignments].detach() + out_res_shs
        full_opacity = torch.clamp(
            compressed_agg_opacity[sorted_node_assignments].detach() + out_res_opacity, 0, 1
        )
        # print("-------------HERE------------------------------")
        # print(compressed_agg_mean3D.min(), compressed_agg_mean3D.max())
        # print(compressed_agg_covariance.min(), compressed_agg_covariance.max())
        # print(compressed_agg_opacity.min(), compressed_agg_opacity.max())
        # print(compressed_agg_shs.min(), compressed_agg_shs.max())
        # print("---------------------------------------------------")
        # print(full_covariance.min(), full_covariance.max())
        # print(full_position.min(), full_position.max())
        # print(full_opacity.min(), full_opacity.max())
        # print(full_shs.min(), full_shs.max())
        # print(out_res_pos.min(), out_res_pos.max())
        # print(out_res_opacity.min(), out_res_opacity.max())
        # print(out_res_shs.min(), out_res_shs.max())
        # breakpoint()
        ########################################################################

        intra_size = compressor.calculate_entropy(intra_likelihoods)
        inter_size = compressor.calculate_entropy(inter_likelihoods)
        total_bits = intra_size + inter_size

        entropy_loss = (intra_size / intra_num_coeffs) + (inter_size / res_num_coeffs)

        render_intra_pkg = render(
            viewpoint_cam,
            compressed_agg_mean3D,
            compressed_agg_opacity,
            compressed_agg_covariance,
            compressed_agg_shs,
            gaussians.active_sh_degree,
            gaussians.max_sh_degree,
            pipeline,
            bg,
        )
        image_intra = render_intra_pkg["render"]

        render_inter_pkg = render(
            viewpoint_cam,
            full_position,
            full_opacity,
            full_covariance,
            full_shs,
            gaussians.active_sh_degree,
            gaussians.max_sh_degree,
            pipeline,
            bg,
        )
        image_inter = render_inter_pkg["render"]

        if iteration % 2000 == 0:
            # Save image for visualization
            logger.info(f"Intra Size: {(intra_size / (8 * (2**20) * subsample_fraction)):.{5}f} MB")
            logger.info(f"Inter Size: {(inter_size / (8 * (2**20) * subsample_fraction)):.{5}f} MB")
            logger.info(f"Node assignments size: {node_assignment_size / (2**20):.{5}f} MB")
            # logger.info(f"Aggregated means3D size: {agg_means_size / (2**20):.{5}f} MB")
            print("Intra Scale:", compressor.intra_compressor.scale[:, :10])
            print("Inter scale:", compressor.inter_compressor.scale[:, :10])
            logger.save_image(
                torch.clamp(image_intra, 0.0, 1.0),
                f"{logger.rendering_dir}/intra_{iteration}_{viewpoint_cam.image_name}.png",
            )
            logger.save_image(
                torch.clamp(image_inter, 0.0, 1.0),
                f"{logger.rendering_dir}/inter_{iteration}_{viewpoint_cam.image_name}.png",
            )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1_intra = l1_loss(image_intra, gt_image)
        Ll1_inter = l1_loss(image_inter, gt_image)
        loss = (1.0 - optimization.lambda_dssim) * Ll1_inter + optimization.lambda_dssim * (
            1.0 - ssim(image_inter, gt_image)
        ) + (1.0 - optimization.lambda_dssim) * Ll1_intra + optimization.lambda_dssim * (
            1.0 - ssim(image_intra, gt_image)
        )

        # loss = loss + optimization.lambda_opacity * torch.abs(gaussians.get_opacity).mean()
        # loss = loss + optimization.lambda_scale * torch.abs(gaussians.get_scaling).mean()

        compression_loss = entropy_lambdas[entropy_lambda_level] * entropy_loss
        pos_loss = l1_loss(full_position, gaussians.get_xyz[sorted_indices])
        covariance_loss = l1_loss(full_covariance, gaussians.get_covariance()[sorted_indices]) + \
            l1_loss(intra_covariance, covariance[intra_indices])
        sh_loss = l1_loss(full_shs, gaussians.get_features[sorted_indices]) + \
            l1_loss(intra_compressed_attrs[:, 7:].view(intra_compressed_attrs.shape[0], -1, 3), shs[intra_indices])
        opacity_loss = l1_loss(full_opacity, gaussians.get_opacity[sorted_indices]) + \
            l1_loss(intra_compressed_attrs[:, 6:7], opacity[intra_indices])

        loss: torch.Tensor = (
            loss
            + compression_loss
            + (
                optimization.lambda_rec * pos_loss
                + optimization.lambda_rec * covariance_loss
                + optimization.lambda_rec * sh_loss
                + optimization.lambda_rec * opacity_loss
            )
        )

        aux_loss = compressor.aux_loss()

        loss.backward()
        aux_loss.backward()

        # print("------------LOSS------------")
        # print(loss)
        # print(compressor.intra_compressor.scale.grad)
        # print(compressor.inter_compressor.hyper_scale.grad)
        # print(compressor.inter_compressor.scale.grad)
        # print(compressor.inter_compressor.hyper_scale.grad)
        # breakpoint()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_loss_for_pos = 0.4 * pos_loss.item() + 0.6 * ema_loss_for_pos
            ema_loss_for_covariance = 0.4 * covariance_loss.item() + 0.6 * ema_loss_for_covariance
            ema_loss_for_sh = 0.4 * sh_loss.item() + 0.6 * ema_loss_for_sh
            ema_loss_for_opacity = 0.4 * opacity_loss.item() + 0.6 * ema_loss_for_opacity

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Pos": f"{ema_loss_for_pos:.{7}f}",
                        "Cov": f"{ema_loss_for_covariance:.{7}f}",
                        "SH": f"{ema_loss_for_sh:.{7}f}",
                        "Opacity": f"{ema_loss_for_opacity:.{7}f}",
                        "Size": f"{(total_bits / (8 * (2**20) * subsample_fraction)).item():.{5}f} MB",
                        "Subsample": f"{subsample_fraction:.{3}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == optimization.iterations:
                progress_bar.close()

            losses = {
                "l1": Ll1_inter.item() + Ll1_intra.item(),
                # "l1": -1.0,
                "mask": 0.0,
                "compression": (
                    compression_loss.item() if iteration >= COMPRESSION_TRAIN_ITER_FROM else 0.0
                ),
                "total": loss.item(),
                "estimated_bits": (total_bits / (8 * (2**20) * subsample_fraction)).item(),
            }

            if (iteration in optimization.test_iterations) and (
                iteration >= COMPRESSION_TRAIN_ITER_FROM
            ):
                logger.info(f"Test cut depth: {chosen_depth} -> {max_depth}")

                def eval_compressed(viewpoint, gaussians, pipeline, bg):
                    losses["test_bits"] = 0.0
                    eb = copy.deepcopy(compressor)
                    eb.update()

                    chosen_depth, max_depth = choose_min_max_depth(gaussians.get_xyz)

                    # Calculate the hierarchy just once, never update Gaussians
                    (mean3D, covariance, shs, opacity, node_assignment) = eb.get_cut_attributes(
                        gaussians, chosen_depth, max_depth
                    )
                    covariance_L = apply_cholesky(covariance)

                    # Sort the node assignments for more efficient assignment compression
                    sorted_node_assignments, sorted_indices = torch.sort(node_assignment)

                    intra_input = torch.cat(
                        (
                            covariance_L,
                            opacity,
                            shs.view(-1, 3 * ((dataset.sh_degree + 1) ** 2)),
                        ),
                        dim=1,
                    )

                    intra_compress_result_dict = eb.compress_intra(
                        intra_input, level=entropy_lambda_level,
                    )
                    losses["test_bits"] += sum(intra_compress_result_dict["sizes"])

                    print(f"Intra sizes: {sum(intra_compress_result_dict['sizes']):.{5}f} MB")

                    decompressed_attrs = eb.decompress_intra(
                        intra_compress_result_dict["bitstrings"],
                        size=intra_compress_result_dict["shapes"],
                        level=intra_compress_result_dict["level"],
                    )

                    agg_test_position = mean3D
                    agg_test_covariance = apply_inverse_cholesky(decompressed_attrs[:, :6])
                    agg_test_opacity = torch.clamp(decompressed_attrs[:, 6:7], 0.0, 1.0)
                    agg_test_shs = decompressed_attrs[:, 7:].view(
                        decompressed_attrs.shape[0], -1, 3
                    )

                    res_position, res_covariance, res_shs, res_opacity = (
                        eb.get_residuals(
                            gaussians.get_xyz[sorted_indices],
                            gaussians.get_covariance()[sorted_indices],
                            gaussians.get_features[sorted_indices],
                            gaussians.get_opacity[sorted_indices],
                            agg_test_position[sorted_node_assignments],
                            agg_test_covariance[sorted_node_assignments],
                            agg_test_shs[sorted_node_assignments],
                            agg_test_opacity[sorted_node_assignments]
                        )
                    )

                    inter_input = torch.cat(
                        (
                            res_position,
                            res_covariance,
                            res_opacity,
                            res_shs.view(-1, 3 * ((dataset.sh_degree + 1) ** 2)),
                        ),
                        dim=1,
                    )

                    inter_compress_result_dict = eb.compress_inter(
                        inter_input, level=entropy_lambda_level,
                    )
                    losses["test_bits"] += sum(inter_compress_result_dict["sizes"])

                    print(f"Inter sizes: {inter_compress_result_dict['sizes'][0]:.{5}f} MB")

                    compressed_attrs = eb.decompress_inter(
                        inter_compress_result_dict["bitstrings"],
                        size=inter_compress_result_dict["shapes"],
                        level=inter_compress_result_dict["level"],
                    )

                    test_position = (
                        agg_test_position[sorted_node_assignments] + compressed_attrs[:, :3]
                    )
                    # test_covariance = (
                    #     apply_rotation_and_scaling(
                    #         agg_test_covariance[sorted_node_assignments],
                    #         torch.abs(compressed_attrs[:, 3:6]),
                    #         torch.nn.functional.normalize(compressed_attrs[:, 6:10]),
                    #     )
                    # )
                    test_covariance = make_psd(
                        agg_test_covariance[sorted_node_assignments] + compressed_attrs[:, 3:9]
                    )
                    test_opacity = torch.clamp(
                        agg_test_opacity[sorted_node_assignments] + compressed_attrs[:, 9:10],
                        0.0, 1.0
                    )
                    test_shs = agg_test_shs[sorted_node_assignments] + compressed_attrs[
                        :, 10:
                    ].view(compressed_attrs.shape[0], -1, 3)

                    return render(
                        viewpoint,
                        test_position,
                        test_opacity,
                        test_covariance,
                        test_shs,
                        gaussians.active_sh_degree,
                        gaussians.max_sh_degree,
                        pipeline,
                        bg,
                    )

                testing_function = eval_compressed

            else:
                testing_function = gaussian_render
                losses["test_bits"] = losses["estimated_bits"]

            # Log and save
            logger.training_report(
                iteration,
                losses,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                optimization.test_iterations,
                scene,
                testing_function,
                pipeline,
                background,
            )

            if iteration in optimization.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # if (
            #     iteration < optimization.densify_until_iter
            #     and iteration > optimization.densify_from_iter
            #     and iteration % optimization.densification_interval == 0
            # ):
            #     dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
            #     gaussians.relocate_gs(dead_mask=dead_mask)
            #     gaussians.add_new_gs(cap_max=dataset.cap_max)

            # Optimizer step
            if iteration < optimization.iterations:
                # gaussians.optimizer.step()
                # gaussians.optimizer.zero_grad(set_to_none=True)

                # L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                # actual_covariance = L @ L.transpose(1, 2)

                # def op_sigmoid(x, k=100, x0=0.995):
                #     return 1 / (1 + torch.exp(-k * (x - x0)))

                # noise = (
                #     torch.randn_like(gaussians._xyz)
                #     * (op_sigmoid(1 - gaussians.get_opacity))
                #     * optimization.noise_lr
                #     * xyz_lr
                # )
                # noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                # gaussians._xyz.add_(noise)

                compressor.optimizer.step()
                compressor.optimizer.zero_grad(set_to_none=True)
                compressor.aux_optimizer.step()
                compressor.aux_optimizer.zero_grad(set_to_none=True)

            # if iteration in optimization.radsplat_prune_at:
            #     max_alphas = get_max_alphas_over_training_cameras(gaussians, scene, pipeline)
            #     gaussians.radsplat_prune(max_alphas, optimization.radsplat_prune_threshold)

            if iteration in optimization.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), compressor.capture(), iteration),
                    scene.model_path + "/checkpoints/chkpnt" + str(iteration) + ".pth",
                )


def get_max_alphas_over_training_cameras(gaussians, scene, pipeline):
    # sum_alphas = torch.zeros((self.gaussians._xyz.shape[0], 1), device="cuda")
    max_alphas = torch.zeros((gaussians._xyz.shape[0], 1), device="cuda")
    sum_gaussian_counts = torch.zeros((gaussians._xyz.shape[0]), device="cuda")
    for viewpoint_cam in scene.getTrainCameras():
        retrieval_pkg = get_max_alphas(viewpoint_cam, gaussians, pipeline)
        gaussian_alphas, gaussian_count, radii = (
            retrieval_pkg["gaussian_alphas"],
            retrieval_pkg["gaussian_counts"],
            retrieval_pkg["radii"],
        )

        sum_gaussian_counts += gaussian_count
        max_alphas = torch.max(max_alphas, gaussian_alphas)

    # avg_alphas = sum_alphas / sum_gaussian_counts.unsqueeze(1)
    nan_mask = torch.isnan(max_alphas)
    max_alphas[nan_mask] = 0.0
    return max_alphas
