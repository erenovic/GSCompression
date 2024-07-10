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

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from config.build_config_spaces import ConfigReader
from lpipsPyTorch import lpips
from training.log_training import Logger
from training.training_utils import set_seeds
from utils.image_utils import psnr
from utils.loss_utils import ssim


def readImages(
    renders_dir: str, gt_dir: str
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Reads the images from the given directories and returns them as tensors.

    Args:
        renders_dir (str): The directory containing the rendered images.
        gt_dir (str): The directory containing the ground truth images.

    Returns:
        list: A list of tensors containing the rendered images.
        list: A list of tensors containing the ground truth images.
        list: A list of image names.
    """
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(folder_names: str, gt_dir: str, logger: Logger, test_dir: str):
    """Evaluates the rendered images in the given directories using SSIM, PSNR and LPIPS.

    Args:
        folder_names (str): The directory containing the rendered images.
        gt_dir (str): The directory containing the ground truth images.
        logger (Logger): The logger instance to log the results.
        test_dir (str): The directory to save the evaluation results.
    """

    full_dict = {}
    per_view_dict = {}
    logger.info("")

    for renders_dir in folder_names:

        renders, gts, image_names = readImages(renders_dir, gt_dir)

        full_dict[str(renders_dir)] = {}
        per_view_dict[str(renders_dir)] = {}

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

        logger.info("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("")

        full_dict[str(renders_dir)].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            }
        )
        per_view_dict[str(renders_dir)].update(
            {
                "SSIM": {
                    name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                },
                "PSNR": {
                    name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                },
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            }
        )

    with open(test_dir + "/results.json", "w") as fp:
        json.dump(full_dict, fp, indent=True)
    with open(test_dir + "/per_view.json", "w") as fp:
        json.dump(per_view_dict, fp, indent=True)


def render_sets(
    dataset: Namespace,
    iteration: int,
    pipeline: Namespace,
    skip_train: bool,
    skip_test: bool,
    logger,
    compressor: str | None = None,
):
    """Renders the train and test sets and evaluates the results.

    Args:
        dataset (Namespace): The dataset configuration.
        iteration (int): The iteration to load the model from.
        pipeline (Namespace): The pipeline configuration.
        skip_train (bool): If True, the train set will not be rendered.
        skip_test (bool): If True, the test set will not be rendered.
        logger: The logger instance to log the results.
        compressor (str | None): The compressor to use for testing.
    """

    with torch.no_grad():
        # Set background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        os.makedirs(f"{dataset.model_path}/test_renderings", exist_ok=True)

        if not skip_train:
            logger.info("Train set results:")
            train_dir = f"{dataset.model_path}/test_renderings/train"
            os.makedirs(train_dir, exist_ok=True)

            num_gaussians, folder_names, gts_folder = render_set(
                dataset.model_path + "/test_renderings",
                "train",
                dataset,
                GaussianModel,
                iteration,
                pipeline,
                background,
                compressor,
            )
            logger.info(f"  Number of Gaussians : {num_gaussians}")

            evaluate(folder_names, gts_folder, logger, train_dir)

        if not skip_test:
            logger.info("Test set results:")
            test_dir = f"{dataset.model_path}/test_renderings/test"
            os.makedirs(test_dir, exist_ok=True)

            num_gaussians, folder_names, gts_folder = render_set(
                dataset.model_path + "/test_renderings",
                "test",
                dataset,
                GaussianModel,
                iteration,
                pipeline,
                background,
                compressor,
            )
            logger.info(f"  Number of Gaussians : {num_gaussians}")

            evaluate(folder_names, gts_folder, logger, test_dir)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file, e.g., ./config/preset_configs/mcmc_gaussian.yaml",
    )

    parser.add_argument("--skip_train", action="store_false", default=True)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--model", type=str, help="Gaussian primitive model type (e.g., base, mcmc, radsplat)"
    )
    parser.add_argument(
        "--model_path", type=str, help="Model folder for the experiment"
    )  # "./output/XXXXXX-X/"
    parser.add_argument(
        "--scene_name",
        type=str,
        help="Scene name for the experiment, e.g., 'tandt/train', 'tandt/truck'...",
    )

    parser.add_argument(
        "--eval",
        action="store_false",
        default=True,
        help="If true, evaluation data used for testing (default behavior)",
    )
    parser.add_argument(
        "--load_iteration",
        type=int,
        help="Loaded checkpoint iteration inside the checkpoints folder in model_path",
    )

    parser.add_argument(
        "--compressor",
        type=str,
        default=None,
        help="Compressor to use for testing, e.g., 'meanscale'. By default, no compressor is used.",
    )

    args = parser.parse_args(sys.argv[1:])

    set_seeds()

    # Read config file which contains all the parameters and provided as argument
    reader = ConfigReader(args.config)
    reader.modify_using_args(args)

    optimization = reader.optimization_config
    pipeline = reader.pipeline_config
    dataset = reader.dataset_config

    # Add scene name to the data path
    dataset.data_path = os.path.join(dataset.data_path, args.scene_name)

    # Choose the model to use
    if dataset.model in ["base", "radsplat", "mcmc"]:
        # The evaluation is the same for base, radsplat and mcmc models
        # since they are essentially the same model
        from models.splatting.base_model import GaussianModel
        from testing.testing_base import render_set

    elif dataset.model in ["masked", "masked_mcmc"]:
        # The evaluation is different for masked models since they use a different
        # rendering method (masked rendering which applies the mask before the splatting)
        from models.splatting.masked_model import GaussianModel
        from testing.testing_masked import render_set
    elif dataset.model in ["complete"]:
        pass
    else:
        raise ValueError(f"Unknown model type: {dataset.model}")

    if args.compressor:
        # If a compressor is provided, the evaluation will be done using the compressed model
        # The correct rendering method will be chosen based on the model type
        if dataset.model == "complete":
            print("Using hierarchical/complete compressor")
            from models.splatting.mcmc_model import GaussianModel
            from testing.testing_complete import render_set

        elif dataset.model in ["res_pos"]:
            from models.splatting.base_model import GaussianModel
            from testing.testing_res_pos_compressed import render_set
        elif args.compressor:
            print("Using compressor: ", args.compressor)
            from testing.testing_compressed import render_set
    else:
        print("Using uncompressed model")

    print("Rendering " + dataset.model_path)

    # Updates the config reader instance and saves to the test path
    reader.append_args_to_config_and_save(args, dataset.model_path)

    log_file_path = f"{dataset.model_path}/test_results.log"
    logger = Logger(
        log_file_path=log_file_path,
        wandb_name="",
        args=ConfigReader.concatenate_namespaces(dataset, optimization, pipeline),
    )

    # The compressor and the gaussian model will be initialized inside the render_set function
    render_sets(
        dataset,
        args.load_iteration,
        pipeline,
        args.skip_train,
        args.skip_test,
        logger,
        args.compressor,
    )
