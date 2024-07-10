"""Basic utilities for training."""

import logging
import random
from argparse import Namespace

import numpy as np
import torch


def set_seeds() -> None:
    """Set seeds for reproducibility."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.cuda.manual_seed(0)


def bump_iterations_for_secondary_training(optimization: Namespace, first_iteration: int) -> None:
    """Bump (increment) iterations for secondary training."""

    try:
        extra_iterations = optimization.extra_iterations
    except:
        extra_iterations = 0

    logging.info(f"Additional iterations: {extra_iterations}")
    logging.info("Bumping iterations for secondary training.")
    # Bump the iterations for secondary training
    optimization.checkpoint_iterations = [
        iteration + extra_iterations for iteration in optimization.checkpoint_iterations
    ]
    optimization.test_iterations = [
        iteration + extra_iterations for iteration in optimization.test_iterations
    ]
    optimization.save_iterations = [
        iteration + extra_iterations for iteration in optimization.save_iterations
    ]
    if extra_iterations > 0:
        optimization.iterations = extra_iterations + first_iteration
