from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from models.compression.base_model import BaseCompressor

# from compressai.models import CompressionModel


class MultiRateMeanScaleHyperprior(BaseCompressor):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """Entropy bottleneck compressor.

        Args:
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
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

        self.position_scale = nn.Parameter(
            40 * torch.ones((n_levels, 1), requires_grad=requires_grad), requires_grad=requires_grad
        )
        self.covariance_scale = nn.Parameter(
            40 * torch.ones((n_levels, 1), requires_grad=requires_grad), requires_grad=requires_grad
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

    def scale_inputs(
        self,
        sampled_attributes: torch.Tensor,
        level: int,
    ):
        attrs_to_compress = sampled_attributes

        if level != int(level):
            interp = level - int(level)
            position_scale = (torch.abs(self.position_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.position_scale[int(level) + 1]) ** interp
            )
            covariance_scale = (torch.abs(self.covariance_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.covariance_scale[int(level) + 1]) ** interp
            )
            opacity_scale = (torch.abs(self.opacity_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.opacity_scale[int(level) + 1]) ** interp
            )
            sh_scale = (torch.abs(self.sh_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.sh_scale[int(level) + 1]) ** interp
            )

        else:
            position_scale = torch.abs(self.position_scale[int(level)])
            covariance_scale = torch.abs(self.covariance_scale[int(level)])
            opacity_scale = torch.abs(self.opacity_scale[int(level)])
            sh_scale = torch.abs(self.sh_scale[int(level)])

        attrs_to_compress = attrs_to_compress * torch.cat(
            (
                position_scale.repeat_interleave(3),
                sh_scale.repeat_interleave(3),
                covariance_scale.repeat_interleave(6),
                opacity_scale,
            ),
            dim=0,
        ).unsqueeze(0)

        return attrs_to_compress

    def descale_inputs(self, decoded_attrs, level: int):
        # # Denormalize the decoded attributes
        # decoded_attrs = (decoded_attrs / 100.) * self.std_vals.unsqueeze(-1) + self.mean_vals.unsqueeze(-1)

        if level != int(level):
            interp = level - int(level)
            position_scale = (torch.abs(self.position_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.position_scale[int(level) + 1]) ** interp
            )
            covariance_scale = (torch.abs(self.covariance_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.covariance_scale[int(level) + 1]) ** interp
            )
            opacity_scale = (self.opacity_scale[int(level)] ** (1 - interp)) * (
                torch.abs(self.opacity_scale[int(level) + 1]) ** interp
            )
            sh_scale = (self.sh_scale[int(level)] ** (1 - interp)) * (
                torch.abs(self.sh_scale[int(level) + 1]) ** interp
            )
        else:
            position_scale = torch.abs(self.position_scale[int(level)])
            covariance_scale = torch.abs(self.covariance_scale[int(level)])
            opacity_scale = torch.abs(self.opacity_scale[int(level)])
            sh_scale = torch.abs(self.sh_scale[int(level)])

        decoded_attrs = decoded_attrs / (
            torch.cat(
                (
                    position_scale.repeat_interleave(3),
                    sh_scale.repeat_interleave(3),
                    covariance_scale.repeat_interleave(6),
                    opacity_scale,
                ),
                dim=0,
            ).unsqueeze(0)
        )

        return decoded_attrs

    def normalize_hyper_latent(self, z: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            hyper_scale = (torch.abs(self.hyper_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.hyper_scale[int(level) + 1]) ** interp
            )
        else:
            hyper_scale = torch.abs(self.hyper_scale[int(level)])

        return z * hyper_scale

    def denormalize_hyper_latent(self, z: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            hyper_scale = (torch.abs(self.hyper_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.hyper_scale[int(level) + 1]) ** interp
            )
        else:
            hyper_scale = torch.abs(self.hyper_scale[int(level)])

        return z / hyper_scale

    def forward(
        self,
        res_position: torch.Tensor,
        res_covariance: torch.Tensor,
        res_shs: torch.Tensor,
        res_opacity: torch.Tensor,
        level: int | float,
        subsample_fraction: float,
        max_sh_degree: int = 3,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        if subsample_fraction < 1.0 - 1e-6:
            total_gaussians = res_position.shape[0]
            sample_size = int(total_gaussians * subsample_fraction)
            indices = torch.randperm(total_gaussians)[:sample_size]
        else:
            indices = torch.arange(res_covariance.shape[0])

        res_attrs = torch.cat(
            (
                res_position[indices],
                res_shs[indices].view(len(indices), -1),
                res_covariance[indices],
                res_opacity[indices],
            ),
            dim=1,
        )

        # sampled_attributes = (sampled_attributes - self.mean_vals) / self.std_vals

        attrs_to_compress = self.scale_inputs(res_attrs, level)
        num_coeffs = attrs_to_compress.shape[1]

        z = self.h_a(attrs_to_compress)
        z_normalized = self.normalize_hyper_latent(z, level)
        z_permute = torch.permute(z_normalized, (1, 0)).unsqueeze(0)  # [1, C, N]

        z_hat, z_likelihoods = self.entropy_bottleneck(z_permute)

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)  # [N, C]
        z_denormalized = self.denormalize_hyper_latent(z_hat_permute, level)
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
        compressed_position = decoded_attrs[:, :3]
        compressed_harmonics = decoded_attrs[:, 3 : 3 * ((max_sh_degree + 1) ** 2) + 3]
        compressed_covariance = decoded_attrs[:, 3 * ((max_sh_degree + 1) ** 2) + 3 : -1]
        compressed_features = compressed_harmonics.view(-1, (max_sh_degree + 1) ** 2, 3)
        compressed_opacity = decoded_attrs[:, -1:]

        return (
            (compressed_position, compressed_covariance, compressed_features, compressed_opacity),
            {"y": y_likelihoods, "z": z_likelihoods},
            indices,
            num_coeffs,
            subsample_fraction,
        )

    def compress(
        self,
        res_position: torch.Tensor,
        res_covariance: torch.Tensor,
        res_shs: torch.Tensor,
        res_opacity: torch.Tensor,
        level: int | float,
    ) -> Dict[str, List[bytearray] | List[int]]:
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        res_attrs = torch.cat(
            (res_position, res_shs.view(res_position.shape[0], -1), res_covariance, res_opacity),
            dim=1,
        )
        # attributes = (attributes - self.mean_vals) / self.std_vals

        attrs_to_compress = self.scale_inputs(res_attrs, level)

        z = self.h_a(attrs_to_compress)
        z_normalized = self.normalize_hyper_latent(z, level)
        z_permute = torch.permute(z_normalized, (1, 0)).unsqueeze(0)  # [1, C, N]

        z_strings = self.entropy_bottleneck.compress(z_permute)
        z_hat = self.entropy_bottleneck.decompress(z_strings, [z.shape[0]])

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)
        z_denormalized = self.denormalize_hyper_latent(z_hat_permute, level)
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
            "shapes": [res_position.shape[0]],
            "sizes": [
                (len(string_y) + len(string_z)) / (2**20)
                for string_y, string_z in zip(y_strings, z_strings)
            ],
            "level": level,
        }

    def decompress(
        self, strings: List[bytearray], size: List[int], max_sh_degree: int, level: int | float
    ) -> Dict[str, torch.Tensor]:
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        z_hat = self.entropy_bottleneck.decompress(strings[1], size)

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)
        z_denormalized = self.denormalize_hyper_latent(z_hat_permute, level)
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
        compressed_position = decoded_attrs[:, :3]
        compressed_harmonics = decoded_attrs[:, 3 : 3 * ((max_sh_degree + 1) ** 2) + 3]
        compressed_covariance = decoded_attrs[:, 3 * ((max_sh_degree + 1) ** 2) + 3 : -1]
        compressed_features = compressed_harmonics.view(-1, (max_sh_degree + 1) ** 2, 3)
        compressed_opacity = decoded_attrs[:, -1:]

        return {
            "position": compressed_position,
            "covariance": compressed_covariance,
            "features": compressed_features,
            "opacity": compressed_opacity,
        }
