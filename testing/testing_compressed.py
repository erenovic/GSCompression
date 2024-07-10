import copy
import glob
import logging
import os
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer.compression_renderer import render
from models.compression.multirate_entropybottleneck import MultiRateEntropyBottleneck
from models.compression.multirate_meanscale_hyperprior import MultiRateMeanScaleHyperprior
from scene import Scene


def render_set(
    model_path, name, dataset, GaussianModel, iteration, pipeline, background, compressor
):
    compression_folder = Path(model_path) / "compression"

    os.makedirs(compression_folder, exist_ok=True)
    os.makedirs(f"{model_path}/{name}", exist_ok=True)
    os.makedirs(f"{model_path}/{name}/gt", exist_ok=True)

    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, shuffle=False)

    views = scene.getTrainCameras() if name == "train" else scene.getTestCameras()

    ckpt = torch.load(
        dataset.model_path + f"/checkpoints/chkpnt{iteration}.pth", map_location="cuda"
    )
    model_iter = ckpt[2]
    gaussians.restore(ckpt[0], training_args=None)

    if compressor == "entropybottleneck":
        compressor = MultiRateEntropyBottleneck(
            dataset,
            M=(3 * ((gaussians.max_sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif compressor == "meanscale":
        compressor = MultiRateMeanScaleHyperprior(
            dataset,
            M=(3 * ((gaussians.max_sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()

    compressor.restore_compressor_params(ckpt[1][:-2])

    folder_names = []

    for level in np.arange(0, len(dataset.lambdas_entropy) - 1 + 0.01, 0.5):
        render_path = os.path.join(model_path, name, f"rendering_{iteration}_{level}")
        gts_path = os.path.join(model_path, name, "gt")

        folder_names.append(render_path)

        os.makedirs(render_path, exist_ok=True)

        torch.cuda.synchronize()
        t0 = time.time()

        compress_all(compressor, gaussians, compression_folder, level)

        encode_duration = (time.time() - t0) / len(views)
        torch.cuda.synchronize()
        t0 = time.time()

        compressed_attrs, xyz, active_sh_degree = decompress_all(dataset, compression_folder, level)

        decode_duration = (time.time() - t0) / len(views)

        logging.info(f"Encode: {encode_duration:.5f} per view for level {level}")
        logging.info(f"Decode: {decode_duration:.5f} per view for level {level}")

        gaussian_params_size = 0
        for file in glob.glob(str(compression_folder / "compressed_gaussian_*.bin")):
            gaussian_params_size += os.path.getsize(file) / (2**20)

        logging.info(
            f"The size of the gaussian params is: {gaussian_params_size} MBytes in level {level}"
        )

        model_size = os.path.getsize(compression_folder / "compressor.pth") / (2**20)
        logging.info(f"The size of the model params is: {model_size} MBytes in level {level}")

        xyz_size = os.path.getsize(compression_folder / "xyz.npz") / (2**20)
        logging.info(f"The size of the xyz is: {xyz_size} MBytes in level {level}")

        logging.info(
            f"Total size: {gaussian_params_size + model_size + xyz_size} MBytes in level {level}"
        )

        t_list = []

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            torch.cuda.synchronize()
            t0 = time.time()

            rendering = render(
                view,
                xyz,
                gaussians.opacity_activation(compressed_attrs["opacity"]),
                gaussians.scaling_activation(compressed_attrs["scaling"]),
                gaussians.rotation_activation(compressed_attrs["rotation"]),
                compressed_attrs["features"],
                active_sh_degree,
                active_sh_degree,
                pipeline,
                background,
            )["render"]

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


def compress_all(compressor, gaussians, folder: Path, level):

    copy_compressor = copy.deepcopy(compressor)

    bitstring_path = folder / "compressed_gaussian.bin"
    model_path = folder / "compressor.pth"
    xyz_path = folder / "xyz.npz"

    compressor.update()
    compressed_result = compressor.compress(gaussians, level=level)

    # 1. Save the compressed gaussian primitives
    for idx, bitstring in enumerate(compressed_result["bitstrings"]):
        cur_bitstring_path = bitstring_path.with_name(f"compressed_gaussian_{idx}.bin")
        with open(cur_bitstring_path, "wb") as file:
            file.write(bitstring[0])

    # 2. Save the compressor
    model_params = copy_compressor.capture_compressor_params(
        compressed_result["shapes"][0], gaussians.active_sh_degree
    )

    torch.save(model_params, model_path)

    # 3. Save the xyz as a pth file (after quantization)
    xyz = gaussians.get_xyz.type(torch.float16).detach().cpu().numpy()
    np.savez_compressed(xyz_path, xyz=xyz)
    # xyz = gaussians.get_xyz.type(torch.float16)
    # torch.save(xyz, xyz_path)


def decompress_all(dataset: Namespace, folder: Path, level):
    bitsring_path = folder / "compressed_gaussian.bin"
    model_path = folder / "compressor.pth"
    xyz_path = folder / "xyz.npz"

    loaded_bitstrings = []

    for compressed_gaussian_path in glob.glob(
        str(bitsring_path.with_name("compressed_gaussian_*.bin"))
    ):
        with open(compressed_gaussian_path, "rb") as file:
            loaded_bitstrings.append([file.read()])

    loaded_model_params = torch.load(model_path)

    xyz = torch.Tensor(np.load(xyz_path)["xyz"]).type(torch.float32).cuda()
    # xyz = torch.load(xyz_path).type(torch.float32).cuda()

    # TODO: Choose the compressor type
    if len(loaded_bitstrings) == 1:
        compressor = MultiRateEntropyBottleneck(
            dataset,
            M=(3 * ((loaded_model_params[-1] + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif len(loaded_bitstrings) == 2:
        compressor = MultiRateMeanScaleHyperprior(
            dataset,
            M=(3 * ((loaded_model_params[-1] + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()

    message = compressor.restore_compressor_params(loaded_model_params)
    print(message)
    compressor.update()

    compressed_attrs = compressor.decompress(
        loaded_bitstrings, [loaded_model_params[-2]], loaded_model_params[-1], level=level
    )

    return compressed_attrs, xyz, loaded_model_params[-1]
