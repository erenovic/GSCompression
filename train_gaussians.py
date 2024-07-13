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
from argparse import ArgumentParser

import torch

from config.build_config_spaces import ConfigReader
from training.log_training import Logger
from training.training_utils import set_seeds

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file, e.g., ./config/preset_configs/mcmc_gaussian.yaml",
    )

    parser.add_argument(
        "--debug_from",
        type=int,
        default=-1,
        help="Not utilized but kept for compatibility with the original codebase",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        default=False,
        help="Not utilized but kept for compatibility with the original codebase",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to train, initialized in config file as well",
    )
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=None,
        help="Test iterations to perform validation, initialized in config file as well",
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=None,
        help="Not utilized but kept for compatibility with the original codebase",
    )
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=None,
        help="Save checkpoints at these iterations, initialized in config file as well",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging, default is not using it",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint .pth file path to load"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        help="Scene name for the experiment, e.g., 'tandt/train', 'tandt/truck'...",
    )
    parser.add_argument(
        "--model", type=str, help="Gaussian primitive model type (e.g., base, mcmc, radsplat)"
    )
    parser.add_argument(
        "--model_path", type=str, help="Model folder for the experiment"
    )  # "./output/XXXXXX-X/"
    parser.add_argument(
        "--eval",
        action="store_false",
        default=True,
        help="If true, evaluation data is not used for training (default behavior)",
    )

    parser.add_argument(
        "--radsplat_prune_at",
        nargs="+",
        type=int,
        default=None,
        help="Apply RadSplat pruning at these levels",
    )

    parser.add_argument(
        "--cap_max", type=int, default=-1, help="MCMC model maximum number of Gaussian primitives"
    )

    args = parser.parse_args(sys.argv[1:])

    # args.save_iterations.append(args.iterations)

    set_seeds()

    reader = ConfigReader(args.config)
    reader.modify_using_args(args)

    optimization = reader.optimization_config
    pipeline = reader.pipeline_config
    dataset = reader.dataset_config
    dataset.data_path = os.path.join(dataset.data_path, args.scene_name)

    print("Optimizing " + dataset.model_path)

    os.makedirs(f"{dataset.model_path}", exist_ok=True)
    os.makedirs(f"{dataset.model_path}/checkpoints", exist_ok=True)
    os.makedirs(f"{dataset.model_path}/training", exist_ok=True)
    os.makedirs(f"{dataset.model_path}/train_renderings", exist_ok=True)
    reader.append_args_to_config_and_save(args, dataset.model_path)

    wandb_name = f"{dataset.model}"
    log_file_path = f"{dataset.model_path}/training/inspection.log"
    logger = Logger(
        log_file_path=log_file_path,
        wandb_name=wandb_name,
        args=ConfigReader.concatenate_namespaces(dataset, optimization, pipeline),
    )

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if dataset.model == "base":
        logger.info("Training base model")
        from training.training_base import training
    elif dataset.model == "masked":
        from training.training_masked import training
    elif dataset.model == "mcmc":
        from training.training_mcmc import training
    elif dataset.model == "masked_mcmc":
        from training.training_masked_mcmc import training
    elif dataset.model == "radsplat":
        from training.training_radsplat import training
    elif dataset.model == "hierarchical":
        raise ValueError(
            "Hierarchical model is not included in this repository.\n"
            "Use mcmc for Gaussian primitives training."
        )
    elif dataset.model == "joint":
        raise ValueError(
            "Joint model is not included in this repository.\n"
            "Use mcmc for Gaussian primitives training."
        )
    elif dataset.model == "latent":
        raise ValueError(
            "Latent model is not included in this repository.\n"
            "Use mcmc for Gaussian primitives training."
        )
    elif dataset.model == "octree":
        raise ValueError(
            "Octree model is not included in this repository.\n"
            "Use mcmc for Gaussian primitives training."
        )
    else:
        raise ValueError(f"Unknown model type: {dataset.model}")

    training(
        dataset,
        optimization,
        pipeline,
        optimization.test_iterations,
        optimization.save_iterations,
        optimization.checkpoint_iterations,
        dataset.checkpoint,
        logger,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
