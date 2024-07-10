import logging
import os
import time

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render
from scene import Scene


def render_set(
    model_path, name, dataset, GaussianModel, iteration, pipeline, background, compressor=None
):
    os.makedirs(f"{model_path}/{name}", exist_ok=True)
    os.makedirs(f"{model_path}/{name}/rendering_{iteration}", exist_ok=True)
    os.makedirs(f"{model_path}/{name}/gt", exist_ok=True)

    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, shuffle=False)

    views = scene.getTrainCameras() if name == "train" else scene.getTestCameras()

    ckpt = torch.load(
        dataset.model_path + f"/checkpoints/chkpnt{iteration}.pth", map_location="cuda"
    )
    model_iter = ckpt[1]
    gaussians.restore(ckpt[0], training_args=None)

    render_path = os.path.join(model_path, name, f"rendering_{iteration}")
    gts_path = os.path.join(model_path, name, "gt")

    folder_names = [render_path]

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize()
        t0 = time.time()

        rendering = render(view, gaussians, pipeline, background)["render"]

        torch.cuda.synchronize()
        t1 = time.time()
        t_list.append(t1 - t0)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    logging.info(f"Test FPS: {fps:.5f}")

    return gaussians.get_xyz.shape[0], folder_names, gts_path
