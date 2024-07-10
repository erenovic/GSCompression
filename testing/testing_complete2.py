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
from models.splatting.hierarchical.hierarchy_utils import choose_min_max_depth
from scene import Scene
from utils.general_utils import apply_cholesky, apply_inverse_cholesky, make_psd, apply_rotation_and_scaling


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
        compressor = CompleteEntropyBottleneck(
            dataset,
            M=(3 * ((gaussians.max_sh_degree + 1) ** 2)) + 4 + 3 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif compressor == "meanscale":
        compressor = CompleteMeanScaleHyperprior(
            dataset,
            M=(3 * ((gaussians.max_sh_degree + 1) ** 2)) + 4 + 3 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()

    compressor.restore_model_params(ckpt[1][:-2])

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

        compressed_attrs, active_sh_degree = decompress_all(
            compressor, dataset, compression_folder, level
        )

        decode_duration = (time.time() - t0) / len(views)

        logging.info(f"Encode: {encode_duration:.5f} per view for level {level}")
        logging.info(f"Decode: {decode_duration:.5f} per view for level {level}")

        intra_params_size = 0
        inter_params_size = 0
        for file in glob.glob(str(compression_folder / "intra_gaussian_*.bin")):
            intra_params_size += os.path.getsize(file) / (2**20)
        for file in glob.glob(str(compression_folder / "inter_gaussian_*.bin")):
            inter_params_size += os.path.getsize(file) / (2**20)

        logging.info(
            f"The size of the intra params is: {intra_params_size} MBytes in level {level}"
        )
        logging.info(
            f"The size of the inter params is: {inter_params_size} MBytes in level {level}"
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
            f"Total size: {intra_params_size + inter_params_size + xyz_size + model_size + node_assignment_size} MBytes in level {level}"
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


def compress_all(compressor, gaussians, folder: Path, level):

    copy_compressor = copy.deepcopy(compressor)

    bitstring_path = folder / "compressed_gaussian.bin"
    model_path = folder / "compressor.pth"
    node_assignment_path = folder / "node_assignments.npz"
    xyz_path = folder / "xyz.npz"

    with torch.no_grad():
        # Calculate the hierarchy just once, never update Gaussians
        chosen_depth, max_depth = choose_min_max_depth(gaussians.get_xyz)

        (mean3D, covariance, shs, opacity, node_assignment) = compressor.get_cut_attributes(
            gaussians, chosen_depth, max_depth
        )
        covariance_L = apply_cholesky(covariance)

        # Sort the node assignments for more efficient assignment compression
        sorted_node_assignments, sorted_indices = torch.sort(node_assignment)

        np.savez_compressed(
            f"{node_assignment_path}",
            node_assignment=sorted_node_assignments.cpu().numpy(),
        )

    compressor.update()

    intra_input = torch.cat(
        (
            covariance_L,
            opacity,
            shs.view(-1, 3 * ((gaussians.max_sh_degree + 1) ** 2)),
        ),
        dim=1,
    )

    intra_compress_result_dict = compressor.compress_intra(
        intra_input, level=level,
    )
    decompressed_attrs = compressor.decompress_intra(
        intra_compress_result_dict["bitstrings"],
        size=intra_compress_result_dict["shapes"],
        level=intra_compress_result_dict["level"],
    )

    agg_test_position = mean3D.clone()
    agg_test_covariance = apply_inverse_cholesky(decompressed_attrs[:, :6])
    agg_test_opacity = torch.clamp(decompressed_attrs[:, 6:7], 0, 1)
    agg_test_shs = decompressed_attrs[:, 7:].view(decompressed_attrs.shape[0], -1, 3)

    res_position, res_scaling, res_rotation, res_shs, res_opacity = compressor.get_residuals(
        gaussians.get_xyz[sorted_indices],
        gaussians.get_covariance()[sorted_indices],
        gaussians.get_features[sorted_indices],
        gaussians.get_opacity[sorted_indices],
        agg_test_position[sorted_node_assignments],
        agg_test_covariance[sorted_node_assignments],
        agg_test_shs[sorted_node_assignments],
        agg_test_opacity[sorted_node_assignments]
    )

    inter_input = torch.cat(
        (
            res_position,
            res_scaling,
            res_rotation,
            res_opacity,
            res_shs.view(-1, 3 * ((gaussians.max_sh_degree + 1) ** 2)),
        ),
        dim=1,
    )

    inter_compress_result_dict = compressor.compress_inter(
        inter_input, level=level,
    )

    # 1. Save the compressed gaussian primitives
    for idx, bitstring in enumerate(intra_compress_result_dict["bitstrings"]):
        cur_bitstring_path = bitstring_path.with_name(f"intra_gaussian_{idx}.bin")
        with open(cur_bitstring_path, "wb") as file:
            file.write(bitstring[0])

    for idx, bitstring in enumerate(inter_compress_result_dict["bitstrings"]):
        cur_bitstring_path = bitstring_path.with_name(f"inter_gaussian_{idx}.bin")
        with open(cur_bitstring_path, "wb") as file:
            file.write(bitstring[0])

    # 2. Save the compressor
    model_params = copy_compressor.capture_model_params(
        intra_compress_result_dict["shapes"][0],
        inter_compress_result_dict["shapes"][0],
        gaussians.active_sh_degree,
    )

    torch.save(model_params, model_path)

    # # 3. Save the xyz as a pth file (after quantization)
    np.savez_compressed(xyz_path, xyz=mean3D.cpu().numpy())


def decompress_all(actual_model, dataset: Namespace, folder: Path, level):
    bitstring_path = folder / "compressed_gaussian.bin"
    model_path = folder / "compressor.pth"
    xyz_path = folder / "xyz.npz"
    node_assignment_path = folder / "node_assignments.npz"

    node_assignments = torch.from_numpy(np.load(node_assignment_path)["node_assignment"]).cuda()

    intra_loaded_bitstrings = []
    inter_loaded_bitstrings = []

    for compressed_gaussian_path in sorted(
        glob.glob(str(bitstring_path.with_name(f"intra_gaussian_*.bin")))
    ):
        with open(compressed_gaussian_path, "rb") as file:
            intra_loaded_bitstrings.append([file.read()])

    for compressed_gaussian_path in sorted(
        glob.glob(str(bitstring_path.with_name(f"inter_gaussian_*.bin")))
    ):
        with open(compressed_gaussian_path, "rb") as file:
            inter_loaded_bitstrings.append([file.read()])

    loaded_model_params = torch.load(model_path)

    mean3D = torch.Tensor(np.load(xyz_path)["xyz"]).type(torch.float32).cuda()

    # TODO: Choose the compressor type
    if len(intra_loaded_bitstrings) == 1:
        compressor = CompleteEntropyBottleneck(
            dataset,
            M=(3 * ((loaded_model_params[-1] + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif len(intra_loaded_bitstrings) == 2:
        compressor = CompleteMeanScaleHyperprior(
            dataset,
            M=(3 * ((loaded_model_params[-1] + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()

    message = compressor.restore_model_params(loaded_model_params)
    compressor.update()

    intra_decompressed_attrs = actual_model.decompress_intra(
        intra_loaded_bitstrings,
        size=[loaded_model_params[2]],
        level=level,
    )

    inter_decompressed_attrs = actual_model.decompress_inter(
        inter_loaded_bitstrings, size=[loaded_model_params[3]], level=level
    )

    agg_test_covariance = apply_inverse_cholesky(intra_decompressed_attrs[:, :6])
    test_opacity = torch.clamp(
        intra_decompressed_attrs[node_assignments, 6:7] + inter_decompressed_attrs[:, 10:11], 
        0, 1
    )
    agg_test_shs = intra_decompressed_attrs[:, 7:].view(intra_decompressed_attrs.shape[0], -1, 3)
    return {
        "position": mean3D[node_assignments] + inter_decompressed_attrs[:, :3],
        "covariance": make_psd(
            apply_rotation_and_scaling(
                agg_test_covariance[node_assignments],
                torch.abs(inter_decompressed_attrs[:, 3:6]),
                torch.nn.functional.normalize(inter_decompressed_attrs[:, 6:10]),
            )
        ),
        "opacity": test_opacity,
        "features": agg_test_shs + inter_decompressed_attrs[:, 11:].view(-1, 3 * ((loaded_model_params[-1] + 1) ** 2)),
    }, loaded_model_params[-1]
