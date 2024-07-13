from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn
from compressai import entropy_models

from models.compression.base_model import BaseCompressor
from models.splatting.hierarchical.hierarchy_utils import (aggregate_gaussians_recursively,
                                                           assign_unique_values, build_octree,
                                                           calculate_weights)

# from compressai.models import CompressionModel


class MultiRateEntropyBottleneck(BaseCompressor):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """Entropy bottleneck compressor.

        Args:
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
        """
        super(MultiRateEntropyBottleneck, self).__init__(dataset)

        self.entropy_bottleneck = entropy_models.EntropyBottleneck(M)
        self.dataset = dataset

        if dataset.compression_fixed_quantization:
            requires_grad = False
        else:
            requires_grad = True

        self.n_quant_levels = n_levels

        self.pos_scale = nn.Parameter(
            500 * torch.ones((n_levels, 1), requires_grad=requires_grad),
            requires_grad=requires_grad,
        )
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
                    self.pos_scale,
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

    def scale_inputs(
        self,
        sampled_attributes: torch.Tensor,
        level: int,
    ):

        attrs_to_compress = sampled_attributes

        if level != int(level):
            interp = level - int(level)
            pos_scale = (torch.abs(self.pos_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.pos_scale[int(level) + 1]) ** interp
            )
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
            pos_scale = torch.abs(self.pos_scale[int(level)])
            scaling_scale = torch.abs(self.scaling_scale[int(level)])
            rotation_scale = torch.abs(self.rotation_scale[int(level)])
            opacity_scale = torch.abs(self.opacity_scale[int(level)])
            sh_scale = torch.abs(self.sh_scale[int(level)])

        attrs_to_compress = attrs_to_compress * torch.cat(
            (
                pos_scale.repeat_interleave(3),
                sh_scale.repeat_interleave(3),
                scaling_scale.repeat_interleave(3),
                rotation_scale.repeat_interleave(4),
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
            pos_scale = (torch.abs(self.pos_scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.pos_scale[int(level) + 1]) ** interp
            )
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
            pos_scale = torch.abs(self.pos_scale[int(level)])
            scaling_scale = torch.abs(self.scaling_scale[int(level)])
            rotation_scale = torch.abs(self.rotation_scale[int(level)])
            opacity_scale = torch.abs(self.opacity_scale[int(level)])
            sh_scale = torch.abs(self.sh_scale[int(level)])

        decoded_attrs = decoded_attrs / (
            torch.cat(
                (
                    pos_scale.repeat_interleave(3),
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
        self,
        res_positions,
        gaussians,
        iteration: int,
        level: int = 0,
        subsample_fraction: float = 0.1,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        (sampled_features, sampled_opacity, sampled_scales, sampled_rotations, indices) = (
            self.sample_gaussians(gaussians, fraction=subsample_fraction)
        )
        sampled_res_positions = res_positions[indices]
        sampled_attributes = torch.cat(
            (
                sampled_res_positions,
                sampled_features.flatten(1, 2),
                sampled_scales,
                sampled_rotations,
                sampled_opacity,
            ),
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
        compressed_res_positions = decoded_attrs[:, :3]
        compressed_harmonics = decoded_attrs[:, 3 : 3 + 3 * ((gaussians.max_sh_degree + 1) ** 2)]
        compressed_scales_rotations = decoded_attrs[
            :, 3 + 3 * ((gaussians.max_sh_degree + 1) ** 2) : -1
        ]

        compressed_features = compressed_harmonics.view(-1, (gaussians.max_sh_degree + 1) ** 2, 3)
        compressed_scales = compressed_scales_rotations[:, :3]
        compressed_rotations = compressed_scales_rotations[:, 3:]

        compressed_opacity = decoded_attrs[:, -1:]

        return (
            (
                compressed_res_positions,
                compressed_opacity,
                compressed_scales,
                compressed_rotations,
                compressed_features,
            ),
            {"z": z_likelihoods},
            indices,
            num_coeffs,
            subsample_fraction,
        )

    def compress(
        self, res_positions, sorted_indices, gaussians, level: int | float
    ) -> Dict[str, List[bytearray] | List[int]]:
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        attributes = torch.cat(
            (
                res_positions,
                gaussians.get_features.flatten(1, 2)[sorted_indices],
                gaussians._scaling[sorted_indices],
                gaussians._rotation[sorted_indices],
                gaussians._opacity[sorted_indices],
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
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        decompressed_attrs = self.entropy_bottleneck.decompress(string[0], size)

        decoded_attrs = self.descale_inputs(decompressed_attrs.squeeze(0).permute(1, 0), level)

        # decoded_attrs = decoded_attrs * self.std_vals + self.mean_vals

        # Split the compressed attributes
        compressed_res_positions = decoded_attrs[:, :3]
        compressed_harmonics = decoded_attrs[:, 3 : 3 + 3 * ((max_sh_degree + 1) ** 2)]
        compressed_scales_rotations = decoded_attrs[:, 3 + 3 * ((max_sh_degree + 1) ** 2) : -1]

        compressed_features = compressed_harmonics.view(-1, (max_sh_degree + 1) ** 2, 3)
        compressed_scales = compressed_scales_rotations[:, :3]
        compressed_rotations = compressed_scales_rotations[:, 3:]

        compressed_opacity = decoded_attrs[:, -1:]

        return {
            "res_positions": compressed_res_positions,
            "features": compressed_features,
            "scaling": compressed_scales,
            "rotation": compressed_rotations,
            "opacity": compressed_opacity,
        }

    def get_cut_attributes(
        self,
        gaussians,
        chosen_depth: int,
        max_depth: int,
    ):
        assert chosen_depth <= max_depth, "Chosen depth must be less than or equal to max depth"

        if chosen_depth == -1:
            return (
                gaussians.get_xyz,
                gaussians.get_features,
                gaussians.get_covariance(),
                gaussians.get_opacity,
                gaussians.active_sh_degree,
            )

        node_assignments = build_octree(gaussians.get_xyz, max_depth)
        mu = gaussians.get_xyz.contiguous()
        sigma = gaussians.get_covariance().contiguous()
        opacity = gaussians.get_opacity.contiguous()
        sh = gaussians.get_features.contiguous()

        weights = calculate_weights(sigma, opacity).contiguous()

        # Node ids were unique for the overall tensor, now we make them unique per depth level
        # gaussian_node_assignments = assign_unique_values(gaussian_node_assignments)
        unique_per_col_gaussian_node_assignments = assign_unique_values(
            node_assignments
        ).contiguous()

        node_xyzs, node_sigma, node_opacity, node_shs = aggregate_gaussians_recursively(
            weights,
            mu,
            sigma,
            opacity,
            sh.view(sh.shape[0], -1),
            unique_per_col_gaussian_node_assignments,
            min_level=chosen_depth,
            max_level=max_depth,
        )
        return (
            node_xyzs,
            node_sigma,
            node_shs.view(node_shs.shape[0], -1, 3),
            node_opacity,
            unique_per_col_gaussian_node_assignments[:, chosen_depth],
        )
