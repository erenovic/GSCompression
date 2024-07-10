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

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from gaussian_renderer import get_max_alphas, network_gui, render
from models.compression.compression_utils import get_size_in_bits
from models.splatting.radsplat_model import GaussianModel
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def training(
    dataset,
    optimization,
    pipeline,
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
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, optimization)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

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

        render_pkg = render(viewpoint_cam, gaussians, pipeline, bg)
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
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == optimization.iterations:
                progress_bar.close()

            losses = {
                "l1": Ll1.item(),
                "mask": 0.0,
                "compression": 0.0,
                "total": loss.item(),
                "estimated_bits": (
                    get_size_in_bits(gaussians._xyz)
                    + get_size_in_bits(gaussians._scaling)
                    + get_size_in_bits(gaussians._rotation)
                    + get_size_in_bits(gaussians.get_features)
                    + get_size_in_bits(gaussians._opacity)
                )
                / (8 * 2**20),
            }

            if iteration in testing_iterations:
                losses["test_bits"] = losses["estimated_bits"]

            # Log and save
            logger.training_report(
                iteration,
                losses,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                pipeline,
                background,
            )
            if iteration in saving_iterations:
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

            if iteration in optimization.radsplat_prune_at:
                max_alphas = get_max_alphas_over_training_cameras(gaussians, scene, pipeline)
                gaussians.radsplat_prune(max_alphas, optimization.radsplat_prune_threshold)

            # Optimizer step
            if iteration < optimization.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
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
