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
from gaussian_renderer.covariance_renderer import render as render
from models.compression.compression_utils import get_size_in_bits
from models.splatting.hierarchical.hierarchy_utils import (
    aggregate_gaussians_recursively,
    assign_unique_values,
    build_octree,
    calculate_weights,
    choose_min_max_depth
)
from models.splatting.mcmc_model import GaussianModel
from scene import Scene
from scene.cameras import Camera
from training.training_utils import bump_iterations_for_secondary_training
from utils.general_utils import (
    build_rotation,
    compute_required_rotation_and_scaling,
    transform_covariance_matrix,
)
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
    compressor,
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
            compressor.restore_model(compressor_params[:-2])
            compressor.restore_optimizer(compressor_params[-2], compressor_params[-1])
        else:
            raise ValueError("Invalid checkpoint format")

        gaussians.restore(model_params, optimization)

        with torch.no_grad():
            compressor.init_normalization_params(gaussians)

        if dataset.freeze_geometry:
            freeze_geometry(gaussians, dataset)

    bump_iterations_for_secondary_training(optimization, first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    COMPRESSION_TRAIN_ITER_FROM = dataset.compression_training_from
    entropy_lambdas = dataset.lambdas_entropy
    print(f"Entropy lambdas: {entropy_lambdas}")

    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_loss_for_pos = 0.0
    ema_loss_for_cov = 0.0
    ema_loss_for_sh = 0.0
    ema_loss_for_opacity = 0.0

    progress_bar = tqdm(range(first_iter, optimization.iterations), desc="Training progress")
    first_iter += 1

    with torch.no_grad():
        # Calculate the hierarchy just once, never update Gaussians
        chosen_depth, max_depth = choose_min_max_depth(gaussians.get_xyz)

        (mean3D, covariance, shs, opacity, node_assignment) = get_cut_attributes(
            None, gaussians, 2.0, chosen_depth, max_depth
        )
        res_position, res_covariance, res_shs, res_opacity = get_residuals(
            gaussians, mean3D, covariance, shs, opacity, node_assignment
        )

        np.savez_compressed(
            f"{dataset.model_path}/node_assignments.npz",
            node_assignment=torch.sort(node_assignment)[0].cpu().numpy(),
        )
        size = os.path.getsize(f"{dataset.model_path}/node_assignments.npz")

        # covariance_transform_parameters = compute_required_rotation_and_scaling(
        #     covariance[node_assignment], gaussians.get_covariance()
        # )

    for iteration in range(first_iter, optimization.iterations + 1):
        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

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

        if iteration >= COMPRESSION_TRAIN_ITER_FROM:
            entropy_lambda_level = np.random.randint(0, len(entropy_lambdas))
            # trial_covariance = torch.zeros((res_covariance.shape[0], 6), device="cuda")
            compressed_attrs, likelihoods, compressed_indices, num_coeffs, subsample_fraction = (
                compressor(
                    res_position.detach(),
                    res_covariance.detach(),
                    res_shs.detach(),
                    res_opacity.detach(),
                    level=entropy_lambda_level,
                    subsample_fraction=dataset.compression_subsample_fraction,
                )
            )
            num_coeffs = len(compressed_indices) * num_coeffs

            with torch.autograd.set_detect_anomaly(True):
                out_res_pos = res_position.clone()
                out_res_shs = res_shs.clone()
                out_res_opacity = res_opacity.clone()
                out_res_covariance = res_covariance.clone()

                out_res_pos[compressed_indices] = compressed_attrs[0]
                out_res_shs[compressed_indices] = compressed_attrs[2]
                out_res_opacity[compressed_indices] = compressed_attrs[3]
                full_position = mean3D[node_assignment] + out_res_pos

                # full_covariance = gaussians.get_covariance()
                # transformed_covariances = transform_covariance_matrix(
                #     compressed_attrs[1], covariance[node_assignment][compressed_indices]
                # )
                # full_covariance[compressed_indices] = transformed_covariances

                # non_pos_semidef_compressed = covariance[node_assignment][compressed_indices] + compressed_attrs[1]
                # unpacked_compressed = unpack_lower_triangular_to_full(non_pos_semidef_compressed)
                # pos_semidef_covariances = project_to_positive_semidefinite(unpacked_compressed)
                # lower_pos_semidef_covariances = pack_full_to_lower_triangular(pos_semidef_covariances)
                # out_res_covariance[compressed_indices] = lower_pos_semidef_covariances

                full_covariance = covariance[node_assignment] + out_res_covariance
                full_shs = shs[node_assignment] + out_res_shs
                full_opacity = opacity[node_assignment] + out_res_opacity

            render_pkg = render(
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
            total_bits = compressor.calculate_entropy(likelihoods)

            entropy_loss = total_bits / num_coeffs

        else:
            render_pkg = gaussian_render(viewpoint_cam, gaussians, pipeline, bg)
            total_bits = torch.Tensor(
                [
                    get_size_in_bits(gaussians.get_opacity)
                    + get_size_in_bits(gaussians.get_scaling)
                    + get_size_in_bits(gaussians.get_rotation)
                    + get_size_in_bits(gaussians.get_features)
                    + get_size_in_bits(gaussians.get_xyz)
                ]
            )
            subsample_fraction = 1.0

        image = render_pkg["render"]

        if iteration % 2000 == 0:
            # Save image for visualization
            print("Number of Gaussians:", gaussians._xyz.shape[0])
            print("Context Gaussians:", mean3D.shape[0])
            print("Position scale:", compressor.position_scale[:, 0])
            print("Covariance scale:", compressor.covariance_scale[:, 0])
            print("Opacity scale:", compressor.opacity_scale[:, 0])
            print("SH scale:", compressor.sh_scale[:, :6])
            logger.save_image(
                torch.clamp(image, 0.0, 1.0),
                f"{logger.rendering_dir}/octree2_{iteration}_{viewpoint_cam.image_name}.png",
            )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - optimization.lambda_dssim) * Ll1 + optimization.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        # loss = 0.0

        # loss = loss + optimization.lambda_opacity * torch.abs(gaussians.get_opacity).mean()
        # loss = loss + optimization.lambda_scale * torch.abs(gaussians.get_scaling).mean()

        if iteration >= COMPRESSION_TRAIN_ITER_FROM:
            compression_loss = entropy_lambdas[entropy_lambda_level] * entropy_loss
            pos_loss = l2_loss(compressed_attrs[0], res_position[compressed_indices])
            cov_loss = l2_loss(compressed_attrs[1], res_covariance[compressed_indices])
            sh_loss = l2_loss(compressed_attrs[2], res_shs[compressed_indices])
            opacity_loss = l2_loss(compressed_attrs[3], res_opacity[compressed_indices])
            loss = (
                loss
                + compression_loss
                + (
                    optimization.lambda_rec * pos_loss
                    + optimization.lambda_rec * cov_loss
                    + optimization.lambda_rec * sh_loss
                    + optimization.lambda_rec * opacity_loss
                )
            )

            aux_loss = compressor.aux_loss()

        loss.backward()
        aux_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_loss_for_pos = 0.4 * pos_loss.item() + 0.6 * ema_loss_for_pos
            ema_loss_for_cov = 0.4 * cov_loss.item() + 0.6 * ema_loss_for_cov
            ema_loss_for_sh = 0.4 * sh_loss.item() + 0.6 * ema_loss_for_sh
            ema_loss_for_opacity = 0.4 * opacity_loss.item() + 0.6 * ema_loss_for_opacity

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Pos Loss": f"{ema_loss_for_pos:.{7}f}",
                        "Cov Loss": f"{ema_loss_for_cov:.{7}f}",
                        "SH Loss": f"{ema_loss_for_sh:.{7}f}",
                        "Opacity Loss": f"{ema_loss_for_opacity:.{7}f}",
                        "Size": f"{(total_bits / (8 * (2**20) * subsample_fraction)).item():.{5}f} MB",
                        "Subsample": f"{subsample_fraction:.{3}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == optimization.iterations:
                progress_bar.close()

            losses = {
                "l1": Ll1.item(),
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

                def eval_compressed(viewpoint, gaussians, pipeline, bg):
                    eb = copy.deepcopy(compressor)
                    eb.update(update_quantiles=True)

                    chosen_depth, max_depth = choose_min_max_depth(gaussians.get_xyz)

                    (mean3D, covariance, shs, opacity, node_assignment) = get_cut_attributes(
                        None, gaussians, 2.0, chosen_depth, max_depth
                    )
                    res_position, res_covariance, res_shs, res_opacity = get_residuals(
                        gaussians, mean3D, covariance, shs, opacity, node_assignment
                    )

                    compression_result_dict = eb.compress(
                        res_position.detach(),
                        res_covariance.detach(),
                        res_shs.detach(),
                        res_opacity.detach(),
                        level=entropy_lambda_level,
                    )
                    losses["test_bits"] = compression_result_dict["sizes"][0]

                    decompressed_result = eb.decompress(
                        compression_result_dict["bitstrings"],
                        size=compression_result_dict["shapes"],
                        max_sh_degree=gaussians.max_sh_degree,
                        level=compression_result_dict["level"],
                    )

                    test_position = mean3D[node_assignment] + decompressed_result["position"]
                    test_covariance = covariance[node_assignment] + res_covariance
                    test_shs = shs[node_assignment] + decompressed_result["features"]
                    test_opacities = opacity[node_assignment] + decompressed_result["opacity"]

                    return render(
                        viewpoint,
                        test_position,
                        test_opacities,
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

            # if iteration in optimization.radsplat_prune_at:
            #     max_alphas = get_max_alphas_over_training_cameras(gaussians, scene, pipeline)
            #     gaussians.radsplat_prune(max_alphas, optimization.radsplat_prune_threshold)

            if iteration in optimization.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/checkpoints/chkpnt" + str(iteration) + ".pth",
                )


def get_residuals(
    gaussians: GaussianModel,
    agg_xyz: torch.Tensor,
    agg_covariances: torch.Tensor,
    agg_features: torch.Tensor,
    agg_opacities: torch.Tensor,
    node_assignments: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    residual_position = gaussians.get_xyz - agg_xyz[node_assignments]
    residual_covariance = gaussians.get_covariance() - agg_covariances[node_assignments]
    residual_features = gaussians.get_features - agg_features[node_assignments]
    residual_opacities = gaussians.get_opacity - agg_opacities[node_assignments]
    return residual_position, residual_covariance, residual_features, residual_opacities


def get_cut_attributes(
    camera: Camera,
    gaussians: GaussianModel,
    granularity_threshold: int,
    chosen_depth: int,
    max_depth: int = 20,
):
    assert chosen_depth <= max_depth, "Chosen depth must be less than or equal to max depth"

    if chosen_depth == -1:
        return (
            gaussians.get_xyz,
            gaussians.get_features,
            gaussians.get_covariance(),
            gaussians.get_opacity,
            gaussians.active_sh_degree,
        )

    node_assignments = build_octree(gaussians.get_xyz, max_depth)
    mu = gaussians.get_xyz.contiguous()
    sigma = gaussians.get_covariance().contiguous()
    opacity = gaussians.get_opacity.contiguous()
    sh = gaussians.get_features.contiguous()

    weights = calculate_weights(sigma, opacity).contiguous()

    # Node ids were unique for the overall tensor, now we make them unique per depth level
    # gaussian_node_assignments = assign_unique_values(gaussian_node_assignments)
    unique_per_col_gaussian_node_assignments = assign_unique_values(node_assignments).contiguous()

    node_xyzs, node_sigma, node_opacity, node_shs = aggregate_gaussians_recursively(
        weights,
        mu,
        sigma,
        opacity,
        sh.view(sh.shape[0], -1),
        unique_per_col_gaussian_node_assignments,
        min_level=chosen_depth,
        max_level=max_depth,
    )
    return (
        node_xyzs,
        node_sigma,
        node_shs.view(node_shs.shape[0], -1, 3),
        node_opacity,
        unique_per_col_gaussian_node_assignments[:, chosen_depth],
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
