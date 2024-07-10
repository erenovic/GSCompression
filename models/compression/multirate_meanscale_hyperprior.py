from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from models.compression.base_model import BaseCompressor

# from compressai.models import CompressionModel


class MultiRateMeanScaleHyperprior(BaseCompressor):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """Mean-scale hyperprior compressor.

        Args:
            dataset (Namespace): The dataset configuration.
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
            n_levels (int, optional): Number of levels in the entropy model. Defaults to 1.
        """
        super(MultiRateMeanScaleHyperprior, self).__init__(dataset)

        self.h_a = nn.Sequential(
            nn.Linear(M, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N),
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.h_s = nn.Sequential(
            nn.Linear(N, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M * 3 // 2, 2 * M),
        )

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
        self.hyper_scale = nn.Parameter(
            torch.ones((n_levels, N), requires_grad=requires_grad), requires_grad=requires_grad
        )

    def training_setup(self, optimization_args: Namespace):
        """Set up the optimizer for the entropy bottleneck compressor."""

        l = [
            {
                "params": [p for n, p in self.named_parameters() if not n.endswith(".quantiles")],
                "lr": optimization_args.compressor_lr,
                "name": "params",
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        l_aux = [
            {
                "params": [p for n, p in self.named_parameters() if n.endswith(".quantiles")],
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
        # attrs_to_compress = spherical_harmonics_reshaped

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

    def scale_hyper_latent(self, z: torch.Tensor, level: int):
        """Scale the hyperprior latent variable before compressing.

        Args:
            z (torch.Tensor): The latent variable to compress.
            level (int): The quantization level to use.

        Returns:
            torch.Tensor: The scaled latent variable.
        """
        if level != int(level):
            interp = level - int(level)
            hyper_scale = (torch.abs(self.hyper_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.hyper_scale[int(level) + 1]) ** interp
            )
        else:
            hyper_scale = torch.abs(self.hyper_scale[int(level)])

        return z * hyper_scale

    def descale_hyper_latent(self, z: torch.Tensor, level: int):
        """Descale the hyperprior latent variable after decompressing.

        Args:
            z (torch.Tensor): The latent variable to decompress.
            level (int): The quantization level used.

        Returns:
            torch.Tensor: The descaled latent variable.
        """
        if level != int(level):
            interp = level - int(level)
            hyper_scale = (torch.abs(self.hyper_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.hyper_scale[int(level) + 1]) ** interp
            )
        else:
            hyper_scale = torch.abs(self.hyper_scale[int(level)])

        return z / hyper_scale

    def forward(
        self, gaussians, iteration: int, level: int = 0, subsample_fraction: float = 0.1
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """Forward pass through the mean-scale hyperprior compressor for training

        Args:
            gaussians (GaussianModel): The Gaussian model to compress.
            iteration (int): The current iteration.
            level (int, optional): The quantization level to use. Defaults to 0.
            subsample_fraction (float, optional): The fraction of Gaussians to sample.
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

        z = self.h_a(attrs_to_compress)
        z_normalized = self.scale_hyper_latent(z, level)
        z_permute = torch.permute(z_normalized, (1, 0)).unsqueeze(0)  # [1, C, N]

        z_hat, z_likelihoods = self.entropy_bottleneck(z_permute)

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)  # [N, C]
        z_denormalized = self.descale_hyper_latent(z_hat_permute, level)
        gaussian_params = self.h_s(z_denormalized)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        scales_hat_permute = torch.permute(scales_hat, (1, 0)).unsqueeze(0)
        means_hat_permute = torch.permute(means_hat, (1, 0)).unsqueeze(0)

        encoded_attrs_permute = attrs_to_compress.permute(1, 0).unsqueeze(0)  # [1, C, N]

        compressed_attrs, y_likelihoods = self.gaussian_conditional(
            encoded_attrs_permute, scales_hat_permute, means=means_hat_permute
        )

        decoded_attrs = self.descale_inputs(compressed_attrs.squeeze(0).permute(1, 0), level)

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
            {"y": y_likelihoods, "z": z_likelihoods},
            indices,
            num_coeffs,
            subsample_fraction,
        )

    def compress(self, gaussians, level: int | float) -> Dict[str, List[bytearray] | List[int]]:
        """Actual compression method used for inference time.

        Args:
            gaussians (GaussianModel): The Gaussian model to compress.
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

        z = self.h_a(attrs_to_compress)
        z_normalized = self.scale_hyper_latent(z, level)
        z_permute = torch.permute(z_normalized, (1, 0)).unsqueeze(0)  # [1, C, N]

        z_strings = self.entropy_bottleneck.compress(z_permute)
        z_hat = self.entropy_bottleneck.decompress(z_strings, [z.shape[0]])

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)
        z_denormalized = self.descale_hyper_latent(z_hat_permute, level)
        gaussian_params = self.h_s(z_denormalized)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        scales_hat_permute = torch.permute(scales_hat, (1, 0)).unsqueeze(0)
        means_hat_permute = torch.permute(means_hat, (1, 0)).unsqueeze(0)
        indexes = self.gaussian_conditional.build_indexes(scales_hat_permute)

        encoded_attrs_permute = attrs_to_compress.permute(1, 0).unsqueeze(0)  # [1, C, N]

        y_strings = self.gaussian_conditional.compress(
            encoded_attrs_permute, indexes, means=means_hat_permute
        )

        return {
            "bitstrings": [y_strings, z_strings],
            "shapes": [gaussians.get_xyz.shape[0]],
            "sizes": [
                (len(string_y) + len(string_z)) / (2**20)
                for string_y, string_z in zip(y_strings, z_strings)
            ],
            "level": level,
        }

    def decompress(
        self, strings: List[bytearray], size: List[int], max_sh_degree: int, level: int | float
    ) -> Dict[str, torch.Tensor]:
        """Actual decompression method used for inference time.

        Args:
            string (List[bytearray]): The compressed bitstrings.
            size (List[int]): The size of the compressed bitstrings (number of Gaussians).
            level (int | float): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        z_hat = self.entropy_bottleneck.decompress(strings[1], size)

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)
        z_denormalized = self.descale_hyper_latent(z_hat_permute, level)
        gaussian_params = self.h_s(z_denormalized)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        scales_hat_permute = torch.permute(scales_hat, (1, 0)).unsqueeze(0)
        means_hat_permute = torch.permute(means_hat, (1, 0)).unsqueeze(0)

        indexes = self.gaussian_conditional.build_indexes(scales_hat_permute)
        decompressed_attrs = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat_permute
        )

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
