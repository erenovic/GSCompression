import numpy as np
import torch

from models.splatting.base_model import GaussianModel


def get_subsampling_factor(
    iteration, start_iteration, end_iteration, start_value=0.1, final_value=1.0
):
    """
    Calculate the exponential incrementation for subsampling given an iteration.

    :param iteration: The current iteration.
    :param start_value: The value before exponential incrementation starts (default 0.1).
    :param final_value: The target value at the end of exponential incrementation (default 1.0).
    :param start_iteration: The iteration when exponential incrementation starts (default 5000).
    :param end_iteration: The iteration when exponential incrementation ends (default 18000).
    :return: The value at the given iteration.
    """
    if iteration <= start_iteration:
        return start_value
    elif iteration > end_iteration:
        return final_value
    else:
        # Calculate the exponential growth rate
        b = np.log(final_value / start_value) / (end_iteration - start_iteration)
        # Calculate the value at the given iteration
        value = start_value * np.exp(b * (iteration - start_iteration))
        return value


def calculate_psnr(true_image, test_image):
    """
    Calculate the PSNR between two images.

    Args:
    true_image (numpy.ndarray): The ground truth image (must be float, up to 255).
    test_image (numpy.ndarray): The reconstructed or noisy image (must be float, up to 255).

    Returns:
    float: The PSNR value in decibels (dB).
    """
    # Ensure the input images have the same dimensions
    if true_image.shape != test_image.shape:
        raise ValueError("True image and test image must have the same dimensions")

    # Compute the MSE (Mean Squared Error)
    mse = np.mean((true_image - test_image) ** 2)

    # If MSE is zero, the images are identical, and PSNR is infinite
    if mse == 0:
        return float("inf")

    # Assuming the pixel values range from 0 to 255
    max_pixel = 255.0

    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def get_size_in_bits(tensor: torch.Tensor) -> float:
    """Returns the size of a tensor in bits."""
    return tensor.nelement() * tensor.element_size() * 8


def get_num_bottleneck_channels(gaussians: GaussianModel) -> int:
    channels = torch.permute(
        torch.cat((gaussians._features_dc, gaussians._features_rest), dim=1).flatten(1, 2),
        dims=(1, 0),
    ).shape[0]

    return channels
