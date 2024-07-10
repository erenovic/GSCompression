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
import sys
from random import randint

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render as base_renderer
from gaussian_renderer.compression_renderer import render as compression_renderer
from models.compression.compression_utils import get_size_in_bits
from models.splatting.base_model import GaussianModel
from scene import Scene
from training.training_utils import bump_iterations_for_secondary_training
from utils.loss_utils import l1_loss, ssim


def freeze_geometry(gaussians, optimization):
    if optimization.freeze_geometry_for_compression:
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
        freeze_geometry(gaussians, optimization)

    bump_iterations_for_secondary_training(optimization, first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    COMPRESSION_TRAIN_ITER_FROM = 25_000
    entropy_lambdas = dataset.lambdas_entropy

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, optimization.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, optimization.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipeline.debug = True

        bg = torch.rand((3), device="cuda") if optimization.random_background else background

        if iteration > COMPRESSION_TRAIN_ITER_FROM:
            entropy_lambda_level = np.random.randint(0, len(entropy_lambdas))
            gaussian_params, likelihoods, indices, num_coeffs, subsample_fraction = compressor(
                gaussians, iteration, entropy_lambda_level
            )
            num_coeffs = len(indices) * num_coeffs

            opacities = gaussians.get_opacity.clone()
            opacities[indices] = gaussians.opacity_activation(gaussian_params[0])
            scales = gaussians.get_scaling.clone()
            scales[indices] = gaussians.scaling_activation(gaussian_params[1])
            rotations = gaussians.get_rotation.clone()
            rotations[indices] = gaussians.rotation_activation(gaussian_params[2])
            shs = gaussians.get_features.clone()
            shs[indices] = gaussian_params[3]

            render_pkg = compression_renderer(
                viewpoint_cam,
                gaussians.get_xyz,
                opacities,
                scales,
                rotations,
                shs,
                gaussians.active_sh_degree,
                gaussians.max_sh_degree,
                pipeline,
                bg,
            )

            total_bits = compressor.calculate_entropy(likelihoods)
            entropy_loss = total_bits / num_coeffs

        else:
            render_pkg = base_renderer(viewpoint_cam, gaussians, pipeline, bg)
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

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - optimization.lambda_dssim) * Ll1 + optimization.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        if iteration > COMPRESSION_TRAIN_ITER_FROM:
            loss += entropy_lambdas[entropy_lambda_level] * entropy_loss

        aux_loss = compressor.aux_loss()

        loss.backward()
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
                "mask": 0.0,
                "compression": 0.0,
                "total": loss.item(),
                "estimated_bits": (total_bits / (8 * (2**20) * subsample_fraction)).item(),
            }

            if (iteration in optimization.test_iterations) and (
                iteration > COMPRESSION_TRAIN_ITER_FROM
            ):

                def eval_compressed(viewpoint, gaussians, pipeline, bg):
                    eb = copy.deepcopy(compressor)
                    eb.update(update_quantiles=True)

                    compressed_result = eb.compress(gaussians, level=len(entropy_lambdas) - 1)
                    losses["test_bits"] = compressed_result["sizes"][0]
                    decompressed_result = eb.decompress(
                        compressed_result["bitstrings"],
                        size=compressed_result["shapes"],
                        max_sh_degree=gaussians.max_sh_degree,
                        level=compressed_result["level"],
                    )

                    test_opacities = gaussians.opacity_activation(decompressed_result["opacity"])
                    test_scales = gaussians.scaling_activation(decompressed_result["scaling"])
                    test_rotations = gaussians.rotation_activation(decompressed_result["rotation"])
                    test_shs = decompressed_result["features"]

                    return compression_renderer(
                        viewpoint,
                        gaussians.get_xyz,
                        test_opacities,
                        test_scales,
                        test_rotations,
                        test_shs,
                        gaussians.active_sh_degree,
                        gaussians.max_sh_degree,
                        pipeline,
                        bg,
                    )

                testing_function = eval_compressed

            else:
                testing_function = base_renderer
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

            # Densification
            if iteration < optimization.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (
                    iteration > optimization.densify_from_iter
                    and iteration % optimization.densification_interval == 0
                ):
                    size_threshold = 20 if iteration > optimization.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        optimization.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % optimization.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == optimization.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < optimization.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                compressor.optimizer.step()
                compressor.optimizer.zero_grad(set_to_none=True)

            if iteration in optimization.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), compressor.capture(), iteration),
                    scene.model_path + "/checkpoints/chkpnt" + str(iteration) + ".pth",
                )
