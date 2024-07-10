from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn
from compressai import entropy_models

from models.compression.base_model import BaseCompressor

# from compressai.models import CompressionModel


class MultiRateEntropyBottleneck(BaseCompressor):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """Entropy bottleneck compressor.

        Args:
            dataset (Namespace): The dataset configuration.
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
            n_levels (int, optional): Number of levels in the entropy model. Defaults to 1.
        """
        super(MultiRateEntropyBottleneck, self).__init__(dataset)

        self.entropy_bottleneck = entropy_models.EntropyBottleneck(M)

        if dataset.compression_fixed_quantization:
            requires_grad = False
        else:
            requires_grad = True

        self.n_quant_levels = n_levels

        self.scaling_scale = nn.Parameter(
            20 * torch.ones((n_levels, 1), requires_grad=requires_grad), requires_grad=requires_grad
        )
        self.rotation_scale = nn.Parameter(
            20 * torch.ones((n_levels, 1), requires_grad=requires_grad), requires_grad=requires_grad
        )
        self.opacity_scale = nn.Parameter(
            20 * torch.ones((n_levels, 1), requires_grad=requires_grad), requires_grad=requires_grad
        )
        self.sh_scale = nn.Parameter(
            20 * torch.ones((n_levels, (dataset.sh_degree + 1) ** 2), requires_grad=requires_grad),
            requires_grad=requires_grad,
        )

    def training_setup(self, optimization_args: Namespace):
        """Set up the optimizer for the entropy bottleneck compressor."""
        l = [
            {
                "params": [
                    p
                    for n, p in self.entropy_bottleneck.named_parameters()
                    if not n.endswith(".quantiles")
                ],
                "lr": optimization_args.compressor_lr,
                "name": "params",
            },
            {
                "params": [
                    self.scaling_scale,
                    self.rotation_scale,
                    self.opacity_scale,
                    self.sh_scale,
                ],
                "lr": optimization_args.compressor_lr,
                "name": "scaling_params",
            },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        l_aux = [
            {
                "params": [
                    p
                    for n, p in self.entropy_bottleneck.named_parameters()
                    if n.endswith(".quantiles")
                ],
                "lr": optimization_args.compressor_aux_lr,
                "name": "aux_params",
            }
        ]
        self.aux_optimizer = torch.optim.Adam(l_aux, lr=0.0, eps=1e-15)

    def scale_inputs(self, sampled_attributes: torch.Tensor, level: int):
        """Scale the input attributes to the entropy bottleneck before compressing.

        Args:
            sampled_attributes (torch.Tensor): The attributes to compress.
            level (int): The quantization level to use.

        Returns:
            torch.Tensor: The scaled attributes.
        """

        attrs_to_compress = sampled_attributes

        if level != int(level):
            interp = level - int(level)
            scaling_scale = (torch.abs(self.scaling_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.scaling_scale[int(level) + 1]) ** interp
            )
            rotation_scale = (torch.abs(self.rotation_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.rotation_scale[int(level) + 1]) ** interp
            )
            opacity_scale = (torch.abs(self.opacity_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.opacity_scale[int(level) + 1]) ** interp
            )
            sh_scale = (torch.abs(self.sh_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.sh_scale[int(level) + 1]) ** interp
            )

        else:
            scaling_scale = torch.abs(self.scaling_scale[int(level)])
            rotation_scale = torch.abs(self.rotation_scale[int(level)])
            opacity_scale = torch.abs(self.opacity_scale[int(level)])
            sh_scale = torch.abs(self.sh_scale[int(level)])

        attrs_to_compress = attrs_to_compress * torch.cat(
            (
                sh_scale.repeat_interleave(3),
                scaling_scale.repeat_interleave(3),
                rotation_scale.repeat_interleave(4),
                opacity_scale,
            ),
            dim=0,
        ).unsqueeze(0)

        return attrs_to_compress

    def descale_inputs(self, decoded_attrs: torch.Tensor, level: int):
        """Descale the decoded attributes from the entropy bottleneck.

        Args:
            decoded_attrs (torch.Tensor): The decoded attributes.
            level (int): The quantization level used.

        Returns:
            torch.Tensor: The descaled attributes.
        """
        # # Denormalize the decoded attributes
        # decoded_attrs = (decoded_attrs / 100.) * self.std_vals.unsqueeze(-1) + self.mean_vals.unsqueeze(-1)

        if level != int(level):
            interp = level - int(level)
            scaling_scale = (torch.abs(self.scaling_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.scaling_scale[int(level) + 1]) ** interp
            )
            rotation_scale = (torch.abs(self.rotation_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.rotation_scale[int(level) + 1]) ** interp
            )
            opacity_scale = (self.opacity_scale[int(level)] ** (1 - interp)) * (
                torch.abs(self.opacity_scale[int(level) + 1]) ** interp
            )
            sh_scale = (self.sh_scale[int(level)] ** (1 - interp)) * (
                torch.abs(self.sh_scale[int(level) + 1]) ** interp
            )
        else:
            scaling_scale = torch.abs(self.scaling_scale[int(level)])
            rotation_scale = torch.abs(self.rotation_scale[int(level)])
            opacity_scale = torch.abs(self.opacity_scale[int(level)])
            sh_scale = torch.abs(self.sh_scale[int(level)])

        decoded_attrs = decoded_attrs / (
            torch.cat(
                (
                    sh_scale.repeat_interleave(3),
                    scaling_scale.repeat_interleave(3),
                    rotation_scale.repeat_interleave(4),
                    opacity_scale,
                ),
                dim=0,
            ).unsqueeze(0)
        )

        return decoded_attrs

    def forward(
        self, gaussians, iteration: int, level: int = 0, subsample_fraction: float = 0.1
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """Forward pass of the entropy bottleneck compressor for training.

        Args:
            gaussians (GaussianModel): The gaussians to compress.
            iteration (int): The current iteration.
            level (int, optional): The quantization level to use. Defaults to 0.
            subsample_fraction (float, optional): The fraction of gaussians to sample.
                Defaults to 0.1.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        (sampled_features, sampled_opacity, sampled_scales, sampled_rotations, indices) = (
            self.sample_gaussians(gaussians, fraction=subsample_fraction)
        )
        sampled_attributes = torch.cat(
            (sampled_features.flatten(1, 2), sampled_scales, sampled_rotations, sampled_opacity),
            dim=1,
        )

        # sampled_attributes = (sampled_attributes - self.mean_vals) / self.std_vals

        attrs_to_compress = self.scale_inputs(sampled_attributes, level)
        num_coeffs = attrs_to_compress.shape[1]

        # attrs_to_compress = spherical_harmonics_reshaped
        attrs_to_compress = attrs_to_compress.permute(1, 0).unsqueeze(0)

        z_hat, z_likelihoods = self.entropy_bottleneck(attrs_to_compress)

        decoded_attrs = self.descale_inputs(z_hat.squeeze(0).permute(1, 0), level)

        # decoded_attrs = decoded_attrs * self.std_vals + self.mean_vals

        # Split the compressed attributes
        compressed_harmonics = decoded_attrs[:, : 3 * ((gaussians.max_sh_degree + 1) ** 2)]
        compressed_scales_rotations = decoded_attrs[
            :, 3 * ((gaussians.max_sh_degree + 1) ** 2) : -1
        ]

        compressed_features = compressed_harmonics.view(-1, (gaussians.max_sh_degree + 1) ** 2, 3)
        compressed_scales = compressed_scales_rotations[:, :3]
        compressed_rotations = compressed_scales_rotations[:, 3:]

        compressed_opacity = decoded_attrs[:, -1:]

        return (
            (compressed_opacity, compressed_scales, compressed_rotations, compressed_features),
            {"z": z_likelihoods},
            indices,
            num_coeffs,
            subsample_fraction,
        )

    def compress(self, gaussians, level: int | float) -> Dict[str, List[bytearray] | List[int]]:
        """Actual compression method used for inference time.

        Args:
            gaussians (GaussianModel): The gaussians to compress.
            level (int | float): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        attributes = torch.cat(
            (
                gaussians.get_features.flatten(1, 2),
                gaussians._scaling,
                gaussians._rotation,
                gaussians._opacity,
            ),
            dim=1,
        )
        # attributes = (attributes - self.mean_vals) / self.std_vals
        attrs_to_compress = self.scale_inputs(attributes, level)
        attrs_to_compress = attrs_to_compress.permute(1, 0).unsqueeze(0)

        # attrs_to_compress = attrs_to_compress * self.quantization_scale[level].unsqueeze(0)

        bitstrings = self.entropy_bottleneck.compress(attrs_to_compress)
        return {
            "bitstrings": [bitstrings],
            "shapes": [
                attrs_to_compress[batch_id].shape[-1] for batch_id in range(len(attrs_to_compress))
            ],
            "sizes": [len(string) / (2**20) for string in bitstrings],
            "level": level,
        }

    def decompress(
        self, string: List[bytearray], size: List[int], max_sh_degree: int, level: int | float
    ) -> Dict[str, torch.Tensor]:
        """Actual decompression method used for inference time.

        Args:
            string (List[bytearray]): The compressed bitstrings.
            size (List[int]): The size of the compressed bitstrings (number of Gaussians).
            max_sh_degree (int): The maximum SH degree.
            level (int | float): The quantization level used.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        decompressed_attrs = self.entropy_bottleneck.decompress(string[0], size)

        decoded_attrs = self.descale_inputs(decompressed_attrs.squeeze(0).permute(1, 0), level)

        # decoded_attrs = decoded_attrs * self.std_vals + self.mean_vals

        # Split the compressed attributes
        compressed_harmonics = decoded_attrs[:, : 3 * ((max_sh_degree + 1) ** 2)]
        compressed_scales_rotations = decoded_attrs[:, 3 * ((max_sh_degree + 1) ** 2) : -1]

        compressed_features = compressed_harmonics.view(-1, (max_sh_degree + 1) ** 2, 3)
        compressed_scales = compressed_scales_rotations[:, :3]
        compressed_rotations = compressed_scales_rotations[:, 3:]

        compressed_opacity = decoded_attrs[:, -1:]

        return {
            "features": compressed_features,
            "scaling": compressed_scales,
            "rotation": compressed_rotations,
            "opacity": compressed_opacity,
        }
