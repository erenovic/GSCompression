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
from training.training_utils import bump_iterations_for_secondary_training, set_seeds

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--config", type=str)

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--scene_name", type=str)
    parser.add_argument("--model", type=str)  # model can be "base" or "masked"
    parser.add_argument("--model_path", type=str)  # "./output/XXXXXX-X/"
    parser.add_argument("--eval", action="store_false", default=True)

    parser.add_argument("--cap_max", type=int, default=-1)  # for mcmc model

    # Compression parameters
    parser.add_argument("--compressor", type=str, default=None)
    parser.add_argument("--compressor_lr", type=float, default=None)
    parser.add_argument("--compressor_aux_lr", type=float, default=None)
    parser.add_argument("--compression_fixed_quantization", action="store_true", default=False)
    parser.add_argument("--freeze_geometry", action="store_true", default=False)
    parser.add_argument("--extra_iterations", type=int, default=0)
    parser.add_argument("--compression_training_from", type=int, default=None)

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

    print("Optimizing " + dataset.model_path)

    os.makedirs(f"{dataset.model_path}", exist_ok=True)
    os.makedirs(f"{dataset.model_path}/checkpoints", exist_ok=True)
    os.makedirs(f"{dataset.model_path}/training", exist_ok=True)
    os.makedirs(f"{dataset.model_path}/train_renderings", exist_ok=True)
    reader.append_args_to_config_and_save(args, dataset.model_path)

    wandb_name = f"{dataset.model}_{dataset.lambdas_entropy}"
    log_file_path = f"{dataset.model_path}/training/inspection.log"
    logger = Logger(
        log_file_path=log_file_path,
        wandb_name=wandb_name,
        args=ConfigReader.concatenate_namespaces(dataset, optimization, pipeline),
    )

    # Choose the model to use
    if dataset.model == "base":
        logger.info("Training base model")
        from training.compression.training_base import training
    elif dataset.model == "mcmc":
        from training.compression.training_mcmc import training
    elif dataset.model == "masked":
        raise ValueError(
            "Masked model is not supported for compression training.\n"
            "Use RadSplat pruning model for compression training."
        )
    elif dataset.model == "radsplat":
        raise ValueError(
            "Radsplat model is not supported for compression training.\n"
            "Use mcmc for compression training."
        )
    elif dataset.model == "complete":
        from training.compression.training_complete import training
    elif dataset.model == "hierarchical":
        raise ValueError(
            "Hierarchical model is not included in this repository.\n"
            "Use mcmc for compression training."
        )
    elif dataset.model == "latent":
        # from training.compression.training_latent_mcmc import training
        raise ValueError(
            "Latent model is not included in this repository.\n"
            "Use mcmc for compression training."
        )
    elif dataset.model == "gscodec":
        # from training.compression.training_gscodec import training
        raise ValueError(
            "GSCodec model is not included in this repository.\n"
            "Use mcmc for compression training."
        )
    else:
        raise ValueError(f"Unknown model type: {dataset.model}")

    if args.compressor == "entropybottleneck":
        from models.compression.multirate_entropybottleneck import MultiRateEntropyBottleneck

        compressor = MultiRateEntropyBottleneck(
            dataset,
            M=(3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif args.compressor == "meanscale":
        from models.compression.multirate_meanscale_hyperprior import MultiRateMeanScaleHyperprior

        compressor = MultiRateMeanScaleHyperprior(
            dataset,
            M=(3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif args.compressor == "complete_eb":
        from models.compression.multirate_complete_eb import CompleteEntropyBottleneck

        compressor = CompleteEntropyBottleneck(
            dataset,
            M=(3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 6 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif args.compressor == "complete_ms":
        from models.compression.multirate_complete_ms import CompleteMeanScaleHyperprior

        compressor = CompleteMeanScaleHyperprior(
            dataset,
            M=(3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()
    elif args.compressor == "gscodec":
        # raise ValueError("GSCodec is not working well...")
        from models.compression.gscodec import ConditionalGSCodec

        compressor = ConditionalGSCodec(
            dataset,
            M=(3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 4 + 1,
            N=32,
            n_levels=len(dataset.lambdas_entropy),
        ).cuda()

    training(
        dataset,
        optimization,
        pipeline,
        compressor,
        optimization.test_iterations,
        optimization.save_iterations,
        optimization.checkpoint_iterations,
        dataset.checkpoint,
        logger,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
