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

from gaussian_renderer.covariance_renderer import render
from models.compression.multirate_complete_eb import CompleteEntropyBottleneck
from models.compression.multirate_complete_ms import CompleteMeanScaleHyperprior
from scene import Scene
from utils.general_utils import build_covariance_from_scaling_rotation


def render_set(
    model_path, name, dataset, GaussianModel, iteration, pipeline, background, compressor
):
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, shuffle=False)

    views = scene.getTrainCameras() if name == "train" else scene.getTestCameras()

    ckpt = torch.load(
        dataset.model_path + f"/checkpoints/chkpnt{iteration}.pth", map_location="cuda"
    )
    model_iter = ckpt[2]
    gaussians.restore(ckpt[0], training_args=None)

    if compressor == "complete_eb":
        compressor = CompleteEntropyBottleneck(
            dataset,
            M=(3 * ((gaussians.max_sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif compressor == "complete_ms":
        compressor = CompleteMeanScaleHyperprior(
            dataset,
            M=(3 * ((gaussians.max_sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()

    compressor.restore_model_params(ckpt[1][:-2])

    os.makedirs(f"{model_path}/{name}", exist_ok=True)
    os.makedirs(f"{model_path}/{name}/gt", exist_ok=True)

    folder_names = []

    mean3D, covariance, shs, opacity, node_assignment = compressor.get_cut_attributes(
        gaussians, dataset.min_depth, dataset.max_depth
    )

    sorted_node_assignments, sorted_indices = torch.sort(node_assignment)

    for level in np.arange(0, len(dataset.lambdas_entropy) - 1 + 0.01, 0.5):
        render_path = os.path.join(model_path, name, f"rendering_{iteration}_{level}")
        gts_path = os.path.join(model_path, name, "gt")

        folder_names.append(render_path)

        os.makedirs(render_path, exist_ok=True)
        compression_folder = Path(model_path) / "compression"
        os.makedirs(compression_folder, exist_ok=True)

        torch.cuda.synchronize()
        t0 = time.time()

        compress_all(
            compressor,
            gaussians,
            compression_folder,
            level,
            mean3D,
            covariance,
            shs,
            opacity,
            sorted_node_assignments,
            sorted_indices,
        )

        encode_duration = (time.time() - t0) / len(views)
        torch.cuda.synchronize()
        t0 = time.time()

        compressed_attrs, active_sh_degree = decompress_all(dataset, compression_folder, level)

        decode_duration = (time.time() - t0) / len(views)

        logging.info(f"Encode: {encode_duration:.5f} per view for level {level}")
        logging.info(f"Decode: {decode_duration:.5f} per view for level {level}")

        gaussian_params_size = 0
        for file in glob.glob(str(compression_folder / "intra_gaussian_*.bin")):
            gaussian_params_size += os.path.getsize(file) / (2**20)

        for file in glob.glob(str(compression_folder / "inter_gaussian_*.bin")):
            gaussian_params_size += os.path.getsize(file) / (2**20)

        logging.info(
            f"The size of the gaussian params is: {gaussian_params_size} MBytes in level {level}"
        )

        model_size = os.path.getsize(compression_folder / "compressor.pth") / (2**20)
        logging.info(f"The size of the model params is: {model_size} MBytes in level {level}")

        xyz_size = os.path.getsize(compression_folder / "xyz.npz") / (2**20)
        logging.info(f"The size of the xyz is: {xyz_size} MBytes in level {level}")

        node_assignment_size = os.path.getsize(compression_folder / "node_assignments.npz") / (
            2**20
        )
        logging.info(
            f"The size of the node assignments is: {node_assignment_size} MBytes in level {level}"
        )

        logging.info(
            f"Total size: {gaussian_params_size + model_size + xyz_size + node_assignment_size} MBytes in level {level}"
        )

        t_list = []

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            torch.cuda.synchronize()
            t0 = time.time()

            rendering = render(
                view,
                compressed_attrs["position"],
                compressed_attrs["opacity"],
                compressed_attrs["covariance"],
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


def compress_all(
    compressor: CompleteEntropyBottleneck,
    gaussians,
    folder: Path,
    level,
    mean3D,
    covariance,
    shs,
    opacity,
    sorted_node_assignments,
    sorted_indices,
):

    bitstring_path = folder / "gaussian.bin"  # Name changed later on
    model_path = folder / "compressor.pth"
    xyz_path = folder / "xyz.npz"
    node_assignment_path = folder / "node_assignments.npz"

    copy_compressor = copy.deepcopy(compressor)
    compressor.update()

    # 1. Intra coding of 1st level attributes
    intra_compress_result_dict = compressor.compress_intra(
        (mean3D.detach(), covariance.detach(), shs.detach(), opacity.detach()),
        level=level,
    )
    decompressed_attrs = compressor.decompress_intra(
        intra_compress_result_dict["bitstrings"],
        size=intra_compress_result_dict["shapes"],
        level=intra_compress_result_dict["level"],
    )

    agg_test_shs = decompressed_attrs[:, : 3 * ((gaussians.max_sh_degree + 1) ** 2)].view(
        decompressed_attrs.shape[0], -1, 3
    )
    agg_test_opacity = decompressed_attrs[:, -1:]
    agg_test_covariance = torch.zeros_like(covariance)

    # 2. Calculate residuals for the compressed attributes
    res_position, _, res_shs, res_opacity = compressor.get_residuals(
        gaussians,
        mean3D,
        agg_test_covariance,
        agg_test_shs,
        agg_test_opacity,
        sorted_node_assignments,
        sorted_indices,
    )
    res_covariance = torch.cat(
        [gaussians._scaling[sorted_indices], gaussians._rotation[sorted_indices]],
        dim=1,
    )

    inter_compress_result_dict = compressor.compress_inter(
        (res_position, res_covariance, res_shs, res_opacity),
        level=level,
    )

    # Save the node assignments of Gaussians
    np.savez_compressed(
        node_assignment_path,
        node_assignment=sorted_node_assignments.cpu().numpy(),
    )
    # Save position of positions
    np.savez_compressed(xyz_path, mean3D=mean3D.cpu().numpy())

    # 1. Save the compressed 1st level bins
    for idx, bitstring in enumerate(intra_compress_result_dict["bitstrings"]):
        cur_bitstring_path = bitstring_path.with_name(f"intra_gaussian_{idx}.bin")
        with open(cur_bitstring_path, "wb") as file:
            file.write(bitstring[0])

    # 2. Save the compressed gaussian primitives
    for idx, bitstring in enumerate(inter_compress_result_dict["bitstrings"]):
        cur_bitstring_path = bitstring_path.with_name(f"inter_gaussian_{idx}.bin")
        with open(cur_bitstring_path, "wb") as file:
            file.write(bitstring[0])

    # 3. Save the compressor
    model_params = copy_compressor.capture_model_params(
        intra_compress_result_dict["shapes"][0],
        inter_compress_result_dict["shapes"][0],
        gaussians.active_sh_degree,
    )
    torch.save(model_params, model_path)

    # 4. Save the xyz as a pth file
    np.savez_compressed(xyz_path, xyz=mean3D.detach().cpu().numpy())


def decompress_all(dataset: Namespace, folder: Path, level):
    bitsring_path = folder / "compressed_gaussian.bin"
    model_path = folder / "compressor.pth"
    xyz_path = folder / "xyz.npz"
    node_assignment_path = folder / "node_assignments.npz"

    loaded_intra_bitstrings = []
    loaded_inter_bitstrings = []

    # Load the compressed bitstrings
    for compressed_gaussian_path in glob.glob(str(bitsring_path.with_name("intra_gaussian_*.bin"))):
        with open(compressed_gaussian_path, "rb") as file:
            loaded_intra_bitstrings.append([file.read()])

    for compressed_gaussian_path in glob.glob(str(bitsring_path.with_name("inter_gaussian_*.bin"))):
        with open(compressed_gaussian_path, "rb") as file:
            loaded_inter_bitstrings.append([file.read()])

    # Load model
    loaded_model_params = torch.load(model_path)

    # Load position of 1st level attributes
    xyz = torch.Tensor(np.load(xyz_path)["xyz"]).type(torch.float32).cuda()

    # Load node assignments
    node_assignments = (
        torch.Tensor(np.load(node_assignment_path)["node_assignment"]).type(torch.int32).cuda()
    )

    # TODO: Choose the compressor type
    if len(loaded_intra_bitstrings) == 1:
        compressor = CompleteEntropyBottleneck(
            dataset,
            M=(3 * ((loaded_model_params[-1] + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif len(loaded_intra_bitstrings) == 2:
        compressor = CompleteMeanScaleHyperprior(
            dataset,
            M=(3 * ((loaded_model_params[-1] + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()

    compressor.restore_model_params(loaded_model_params[:2])
    compressor.update()

    intra_compressed_attrs = compressor.decompress_intra(
        loaded_intra_bitstrings, [loaded_model_params[-3]], level=level
    )

    inter_compressed_attrs = compressor.decompress_inter(
        loaded_inter_bitstrings,
        [loaded_model_params[-2]],
        level=level,
    )

    agg_shs = intra_compressed_attrs[:, : 3 * ((loaded_model_params[-1] + 1) ** 2)].view(
        intra_compressed_attrs.shape[0], -1, 3
    )
    agg_opacity = intra_compressed_attrs[:, -1:]
    agg_covariance = torch.zeros_like(intra_compressed_attrs[:, :6])

    test_position = xyz[node_assignments] + inter_compressed_attrs[:, :3]
    test_covariance = agg_covariance[node_assignments] + build_covariance_from_scaling_rotation(
        torch.exp(inter_compressed_attrs[:, 3:6]),
        1.0,
        torch.nn.functional.normalize(inter_compressed_attrs[:, 6:10]),
    )
    test_shs = agg_shs[node_assignments] + inter_compressed_attrs[
        :, 10 : 10 + 3 * ((loaded_model_params[-1] + 1) ** 2)
    ].view(inter_compressed_attrs.shape[0], -1, 3)
    test_opacity = agg_opacity[node_assignments] + inter_compressed_attrs[:, -1:]

    return {
        "position": test_position,
        "covariance": test_covariance,
        "features": test_shs,
        "opacity": test_opacity,
    }, loaded_model_params[-1]
