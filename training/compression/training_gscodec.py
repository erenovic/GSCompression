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
from gaussian_renderer.compression_renderer import render as render
from models.compression.multirate_complete_eb import CompleteEntropyBottleneck
from models.splatting.hierarchical.hierarchy_utils import choose_min_max_depth
from models.splatting.mcmc_model import GaussianModel
from scene import Scene
from scene.cameras import Camera
from training.training_utils import bump_iterations_for_secondary_training
from utils.general_utils import (build_covariance_from_scaling_rotation, build_rotation,
                                 decompose_covariance_matrix, inverse_sigmoid,
                                 rotation_matrix_to_quaternion, unpack_lower_triangular_to_full)
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

        if dataset.freeze_geometry:
            freeze_geometry(gaussians, dataset)

    bump_iterations_for_secondary_training(optimization, first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    entropy_lambdas = dataset.lambdas_entropy
    # subsample_fraction = dataset.compression_subsample_fraction
    subsample_fraction = 1.0
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

        (mean3D, scaling, rotation, covariance, shs, opacity, node_assignment) = (
            compressor.get_cut_attributes(gaussians, chosen_depth, max_depth)
        )

        # Sort the node assignments for more efficient assignment compression
        sorted_node_assignments, sorted_indices = torch.sort(node_assignment)

        np.savez_compressed(
            f"{dataset.model_path}/node_assignments.npz",
            node_assignment=sorted_node_assignments.cpu().numpy(),
        )
        np.savez_compressed(
            f"{dataset.model_path}/aggregated_means3D.npz",
            mean3D=mean3D.cpu().numpy(),
        )
        logger.info(f"Number of Gaussians: {gaussians._xyz.shape[0]}")
        logger.info(f"Number of aggregated Gaussians: {mean3D.shape[0]}")
        node_assignment_size = os.path.getsize(f"{dataset.model_path}/node_assignments.npz")
        logger.info(f"Node assignments size: {node_assignment_size / (2**20):.{5}f} MB")
        agg_means_size = os.path.getsize(f"{dataset.model_path}/aggregated_means3D.npz")
        logger.info(f"Aggregated means3D size: {agg_means_size / (2**20):.{5}f} MB")

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
        subsample_fraction = dataset.compression_subsample_fraction
        subsample_indices = get_subsample_indices(subsample_fraction, mean3D.shape[0])

        x_pred = torch.cat(
            (
                mean3D[sorted_node_assignments[subsample_indices]],
                scaling[sorted_node_assignments[subsample_indices]],
                rotation[sorted_node_assignments[subsample_indices]],
                opacity[sorted_node_assignments[subsample_indices]],
                shs[sorted_node_assignments[subsample_indices]].view(
                    -1, 3 * ((dataset.sh_degree + 1) ** 2)
                ),
            ),
            dim=1,
        )
        x_cur = torch.cat(
            (
                gaussians.get_xyz[sorted_indices[subsample_indices]],
                gaussians.get_scaling[sorted_indices[subsample_indices]],
                gaussians.get_rotation[sorted_indices[subsample_indices]],
                gaussians.get_opacity[sorted_indices[subsample_indices]],
                gaussians.get_features[sorted_indices[subsample_indices]].view(
                    -1, 3 * ((dataset.sh_degree + 1) ** 2)
                ),
            ),
            dim=1,
        )
        out, likelihoods = compressor(x_pred, x_cur, level=entropy_lambda_level)
        ########################################################################

        codec_bits, msg_bits = compressor.calculate_entropy(likelihoods)
        total_bits = codec_bits + msg_bits

        entropy_loss = codec_bits / (x_pred.shape[1] * len(subsample_indices)) + msg_bits / (
            x_pred.shape[1] * len(subsample_indices)
        )

        final_position = gaussians.get_xyz.clone()
        final_scaling = gaussians.get_scaling.clone()
        final_rotation = gaussians.get_rotation.clone()
        final_opacity = gaussians.get_opacity.clone()
        final_shs = gaussians.get_features.clone()

        final_position[sorted_indices[subsample_indices]] = out[:, :3]
        final_scaling[sorted_indices[subsample_indices]] = torch.exp(out[:, 3:6])
        final_rotation[sorted_indices[subsample_indices]] = torch.nn.functional.normalize(
            out[:, 6:10]
        )
        final_opacity[sorted_indices[subsample_indices]] = torch.sigmoid(out[:, 10:11])
        final_shs[sorted_indices[subsample_indices]] = out[:, 11:].view(out.shape[0], -1, 3)

        render_pkg = render(
            viewpoint_cam,
            final_position,
            final_opacity,
            final_scaling,
            final_rotation,
            final_shs,
            gaussians.active_sh_degree,
            gaussians.max_sh_degree,
            pipeline,
            bg,
        )
        image = render_pkg["render"]

        if iteration % 500 == 0:
            # Save image for visualization
            # logger.info(f"Intra Size: {(intra_size / (8 * (2**20) * subsample_fraction)):.{5}f} MB")
            # logger.info(f"Inter Size: {(inter_size / (8 * (2**20) * subsample_fraction)):.{5}f} MB")
            num_params = sum(p.numel() for p in compressor.parameters() if p.requires_grad)
            logger.info(f"Number of parameters: {num_params}")
            logger.info(f"Model size: {num_params * 32 / (8 * (2**20)):.{5}f} MB")
            logger.info(f"Node assignments size: {node_assignment_size / (2**20):.{5}f} MB")
            logger.info(f"Aggregated means3D size: {agg_means_size / (2**20):.{5}f} MB")
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

        compression_loss = entropy_lambdas[entropy_lambda_level] * entropy_loss

        loss: torch.Tensor = loss + compression_loss

        loss += (
            optimization.lambda_rec
            * l2_loss(out[:, :3], gaussians.get_xyz[sorted_indices[subsample_indices]])
            + optimization.lambda_rec
            * l2_loss(
                torch.exp(out[:, 3:6]), gaussians.get_scaling[sorted_indices[subsample_indices]]
            )
            + optimization.lambda_rec
            * l2_loss(
                torch.nn.functional.normalize(out[:, 6:10]),
                gaussians.get_rotation[sorted_indices[subsample_indices]],
            )
            + optimization.lambda_rec
            * l2_loss(
                torch.sigmoid(out[:, 10:11]),
                gaussians.get_opacity[sorted_indices[subsample_indices]],
            )
            + optimization.lambda_rec
            * l2_loss(
                out[:, 11:],
                gaussians.get_features[sorted_indices[subsample_indices]].view(out.shape[0], -1),
            )
        )

        aux_loss = compressor.aux_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
        aux_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
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
                "compression": compression_loss.item(),
                "total": loss.item(),
                "estimated_bits": (total_bits / (8 * (2**20) * subsample_fraction)).item(),
            }

            if iteration in optimization.test_iterations:
                logger.info(f"Test cut depth: {chosen_depth} -> {max_depth}")

                def eval_compressed(viewpoint, gaussians, pipeline, bg):
                    losses["test_bits"] = 0.0
                    eb = copy.deepcopy(compressor)
                    # eb.update()

                    chosen_depth, max_depth = choose_min_max_depth(gaussians.get_xyz)

                    (mean3D, covariance, shs, opacity, node_assignment) = (
                        compressor.get_cut_attributes(gaussians, chosen_depth, max_depth)
                    )

                    scaling, rotation = decompose_covariance_matrix(covariance[node_assignment])

                    x_pred = torch.cat(
                        (
                            mean3D[node_assignment],
                            torch.log(scaling),
                            rotation,
                            inverse_sigmoid(opacity[node_assignment]),
                            shs[node_assignment].view(-1, 3 * ((dataset.sh_degree + 1) ** 2)),
                        ),
                        dim=1,
                    )
                    x_cur = torch.cat(
                        (
                            gaussians.get_xyz,
                            gaussians._scaling,
                            gaussians._rotation,
                            gaussians._opacity,
                            gaussians.get_features.view(-1, 3 * ((dataset.sh_degree + 1) ** 2)),
                        ),
                        dim=1,
                    )
                    out, likelihoods = compressor(
                        x_pred, x_cur, level=entropy_lambda_level, training=False
                    )

                    codec_bits, msg_bits = compressor.calculate_entropy(likelihoods)
                    losses["test_bits"] += codec_bits + msg_bits

                    final_position = out[:, :3]
                    # final_scaling = torch.exp(out[:, 3:6])
                    final_scaling = gaussians.get_scaling
                    final_rotation = torch.nn.functional.normalize(out[:, 6:10])
                    final_opacity = torch.sigmoid(out[:, 10:11])
                    final_shs = out[:, 11:].view(out.shape[0], -1, 3)

                    breakpoint()

                    return render(
                        viewpoint,
                        final_position,
                        final_opacity,
                        final_scaling,
                        final_rotation,
                        final_shs,
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


def get_subsample_indices(subsample_fraction: float, num_gaussians: int):

    if subsample_fraction < 1.0 - 1e-6:
        total_gaussians = num_gaussians
        sample_size = int(total_gaussians * subsample_fraction)
        indices = torch.randperm(total_gaussians)[:sample_size]
    else:
        indices = torch.arange(num_gaussians)

    return indices


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
