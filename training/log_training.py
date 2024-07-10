"""Logger class for training progress and results. It can log to Wandb and to the console."""

import logging
import os
from argparse import Namespace
from typing import Callable, Dict, List, Tuple

import imageio
import numpy as np
import torch

from models.splatting.base_model import GaussianModel
from scene import Scene
from utils.image_utils import psnr

try:
    import wandb

    WANDB_FOUND = True

    wandb.login()
except ImportError:
    WANDB_FOUND = False


class Logger:
    """Logger to log training progress and results to Wandb and logging."""

    def __init__(self, log_file_path: str, wandb_name: str, args: Namespace):
        """Initialize the logging.

        Args:
            log_file_name (str): Model type / Name of the Wandb run at the same time
            args (Namespace): Containing the arguments for the training
        """
        self.args = args
        self.iteration = 0

        # NOTE: Wandb usage is not supported for now!
        self.use_wandb = WANDB_FOUND and args.use_wandb
        if self.use_wandb:
            raise NotImplementedError("Wandb usage is not supported for now!")

        self.rendering_dir = os.path.abspath(
            os.path.dirname(os.path.dirname(log_file_path)) + "/train_renderings"
        )

        self.set_logging(log_file_path)

        # Create Wandb project run
        if WANDB_FOUND and self.use_wandb:
            wandb.init(project="GSCodec", name=wandb_name, config=vars(args))
        else:
            logging.warning("Wandb not used for logging progress")

    def set_logging(self, log_file_path: str) -> None:
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,  # Set the logging level to INFO
            format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log format
            filename=os.path.abspath(log_file_path),  # Specify the log file
            filemode="a",
        )  # Set the file mode to append

    def info(self, message: str) -> None:
        """Log an info message.

        Args:
            message (str): Message to log
        """
        logging.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message (str): Message to log
        """
        logging.warning(message)

    def save_image(self, image: torch.Tensor, path: str) -> None:
        """Save an image to the specified path.

        Args:
            image (torch.Tensor): Image to save
            path (str): Path to save the image
        """
        imageio.imwrite(
            path, (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )

    def training_report(
        self,
        iteration: int,
        losses: Dict[str, float],
        l1_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        elapsed: int,
        testing_iterations: List[int],
        scene: Scene,
        renderFunc: Callable,
        pipeline: Namespace,
        bg: torch.Tensor,
    ) -> None:
        """Report the training progress and results.

        Args:
            iteration (int): Current iteration
            losses (Dict[str, float]): Losses of the current iteration
            l1_loss (Callable): L1 loss function
            elapsed (int): Elapsed time of the current iteration
            testing_iterations (List[int]): Iterations to run the test
            scene (Scene): Scene object
            renderFunc (Callable): Rendering function
            renderArgs (Tuple[Namespace, Namespace]): Rendering arguments
        """
        if self.use_wandb:
            wandb.log(
                {
                    "train/l1_loss": losses["l1"],
                    "train/total_loss": losses["total"],
                    "train/mask_loss": losses["mask"],
                    "train/compression_loss": losses["compression"],
                    "train/estimated_bits": losses["estimated_bits"],
                    "train/elapsed_time": elapsed,
                },
                step=iteration,
            )

        try:
            num_pts = int(scene.gaussians.get_xyz.shape[0])
        except:
            num_pts = int(scene.gaussians._anchor.shape[0])

        if iteration % 1000 == 0:
            logging.info(
                f"[ITER {iteration}]"
                f" L1 Loss: {losses['l1']:.4f}"
                f" Total Loss: {losses['total']:.4f}"
                f" Mask Loss: {losses['mask']:.4f}"
                f" Compression Loss: {losses['compression']:.4f}"
                f" Number of Gaussians: {num_pts}"
                f" Estimated MBytes: {losses['estimated_bits']:.4f}"
                f" Elapsed Time: {elapsed:.2f}"
            )

        # Report test and samples of training set
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = (
                # {"name": "test", "cameras": scene.getTestCameras()},
                {
                    "name": "train",
                    "cameras": [
                        scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                        for idx in range(5, 30, 5)
                    ],
                },
            )

            stats = {
                "test_psnr_sum": 0.0,
                "train_psnr_sum": 0.0,
                "test_l1_loss_sum": 0.0,
                "train_l1_loss_sum": 0.0,
                "num_test_samples": 0,
                "num_train_samples": 0,
            }

            wandb_render_images = []
            wandb_gt_images = []

            for config in validation_configs:
                if config["cameras"] and len(config["cameras"]) > 0:
                    for idx, viewpoint in enumerate(config["cameras"]):
                        render_pkg = renderFunc(viewpoint, scene.gaussians, pipeline, bg)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = viewpoint.original_image
                        if self.use_wandb and (idx < 5):
                            wandb_render_images.append(
                                wandb.Image(
                                    image,
                                    caption="{}_view_{}_iter_{}".format(
                                        config["name"], viewpoint.image_name, iteration
                                    ),
                                )
                            )

                            if iteration == testing_iterations[0]:
                                wandb_gt_images.append(
                                    wandb.Image(
                                        gt_image,
                                        caption="{}_view_{}_gt".format(
                                            config["name"],
                                            viewpoint.image_name,
                                            iteration,
                                        ),
                                    )
                                )

                        img_save_dir = self.rendering_dir

                        imageio.imwrite(
                            f"{img_save_dir}/{iteration}_{config['name']}_{viewpoint.image_name}.png",
                            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                        )

                        stats[f"{config['name']}_l1_loss_sum"] += (
                            l1_loss(image, gt_image).mean().double()
                        )
                        stats[f"{config['name']}_psnr_sum"] += psnr(image, gt_image).mean().double()
                        stats[f"num_{config['name']}_samples"] += 1

                    if self.use_wandb:
                        wandb.log({"images/rendered": wandb_render_images}, step=iteration)

                        if iteration == testing_iterations[0]:
                            wandb.log({"images/gt": wandb_gt_images}, step=iteration)

            num_samples = stats["num_test_samples"] + stats["num_train_samples"]
            psnr_test = (stats["test_psnr_sum"] + stats["train_psnr_sum"]) / num_samples
            l1_test = (stats["test_l1_loss_sum"] + stats["train_l1_loss_sum"]) / num_samples

            logging.info(
                "[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config["name"], l1_test, psnr_test
                )
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "test/l1_loss": l1_test,
                        "test/psnr": psnr_test,
                        "test/true_size": losses["test_bits"],
                        "stats/total_points": num_pts,
                    },
                    step=iteration,
                )

                opacity_table = self.create_hist_line_table(scene.gaussians)

                wandb.log(
                    {
                        "stats/opacity_histogram": wandb.plot.line(
                            opacity_table,
                            "Bin centers",
                            "Gaussian count",
                            title="stats/opacity_histogram",
                        )
                    },
                    step=iteration,
                )

            logging.info(
                f"Test L1 Loss: {l1_test:.4f} PSNR: {psnr_test:.4f}"
                f" True Size: {losses['test_bits']:.4f} MBytes"
                f" Total Points: {num_pts}"
            )

            torch.cuda.empty_cache()

    def create_hist_line_table(self, gaussians: GaussianModel) -> wandb.Table:
        """Create a histogram table for the opacity values of the Gaussian splats.

        Args:
            gaussians (BaseGaussianModel): Gaussian splats to create the histogram from

        Returns:
            wandb.Table: Table containing the histogram data
        """
        hist, bins = np.histogram(gaussians.get_opacity.detach().cpu().numpy())

        bin_centers = (bins[:-1] + bins[1:]) / 2

        data = [[x, y] for (x, y) in zip(bin_centers, hist)]
        table = wandb.Table(data=data, columns=["Bin centers", "Gaussian count"])

        return table
