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
from models.compression.multirate_complete_eb import CompleteEntropyBottleneck
from models.compression.multirate_complete_ms import CompleteMeanScaleHyperprior
from models.splatting.hierarchical.hierarchy_utils import choose_min_max_depth
from scene import Scene
from utils.general_utils import decompose_covariance_matrix


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

        # xyz_size = os.path.getsize(compression_folder / "xyz.npz") / (2**20)
        # logging.info(f"The size of the xyz is: {xyz_size} MBytes in level {level}")

        node_assignment_size = os.path.getsize(compression_folder / "node_assignments.npz") / (
            2**20
        )
        logging.info(
            f"The size of the node assignments is: {node_assignment_size} MBytes in level {level}"
        )

        gaussian_params_size = intra_params_size + inter_params_size
        logging.info(
            f"Total size: {gaussian_params_size + model_size + node_assignment_size} MBytes in level {level}"
        )

        t_list = []

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            torch.cuda.synchronize()
            t0 = time.time()

            rendering = render(
                view,
                compressed_attrs["position"],
                compressed_attrs["opacity"],
                compressed_attrs["scaling"],
                compressed_attrs["rotation"],
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

    with torch.no_grad():
        # Calculate the hierarchy just once, never update Gaussians
        chosen_depth, max_depth = choose_min_max_depth(gaussians.get_xyz)

        (mean3D, scaling, rotation, covariance, shs, opacity, node_assignment) = compressor.get_cut_attributes(
            gaussians, chosen_depth, max_depth
        )

        # Sort the node assignments for more efficient assignment compression
        sorted_node_assignments, sorted_indices = torch.sort(node_assignment)

        np.savez_compressed(
            f"{node_assignment_path}",
            node_assignment=sorted_node_assignments.cpu().numpy(),
        )

        # scaling0, rotation0 = decompose_covariance_matrix(
        #     covariance[: covariance.shape[0] // 2]
        # )
        # scaling1, rotation1 = decompose_covariance_matrix(
        #     covariance[covariance.shape[0] // 2 :]
        # )
        # scaling = torch.cat((scaling0, scaling1), dim=0)
        # rotation = torch.cat((rotation0, rotation1), dim=0)

    compressor.update()

    intra_input = torch.cat(
        (
            mean3D,
            scaling,
            rotation,
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

    agg_test_position = decompressed_attrs[:, :3]
    agg_test_scaling = decompressed_attrs[:, 3:6]
    agg_test_rotation = decompressed_attrs[:, 6:10]
    agg_test_opacity = decompressed_attrs[:, 10:11]
    agg_test_shs = decompressed_attrs[:, 11:].view(decompressed_attrs.shape[0], -1, 3)

    res_position, res_scaling, res_rotation, res_shs, res_opacity = compressor.get_residuals(
        gaussians,
        agg_test_position,
        agg_test_scaling,
        agg_test_rotation,
        agg_test_shs,
        agg_test_opacity,
        sorted_node_assignments,
        sorted_indices,
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
        inter_input,
        level=level,
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
    # xyz = gaussians.get_xyz.type(torch.float16).detach().cpu().numpy()
    # np.savez_compressed(xyz_path, xyz=xyz)
    # # xyz = gaussians.get_xyz.type(torch.float16)
    # # torch.save(xyz, xyz_path)


def decompress_all(actual_model, dataset: Namespace, folder: Path, level):
    bitstring_path = folder / "compressed_gaussian.bin"
    model_path = folder / "compressor.pth"
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

    # xyz = torch.Tensor(np.load(xyz_path)["xyz"]).type(torch.float32).cuda()
    # # xyz = torch.load(xyz_path).type(torch.float32).cuda()

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

    return {
        "position": intra_decompressed_attrs[node_assignments, :3]
        + inter_decompressed_attrs[:, :3],
        "scaling": intra_decompressed_attrs[node_assignments, 3:6]
        + inter_decompressed_attrs[:, 3:6],
        "rotation": intra_decompressed_attrs[node_assignments, 6:10]
        + inter_decompressed_attrs[:, 6:10],
        "opacity": intra_decompressed_attrs[node_assignments, 10:11]
        + inter_decompressed_attrs[:, 10:11],
        "features": (
            intra_decompressed_attrs[node_assignments, 11:] + inter_decompressed_attrs[:, 11:]
        ).view(inter_decompressed_attrs.shape[0], -1, 3),
    }, loaded_model_params[-1]
