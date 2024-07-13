import math
from argparse import Namespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

from models.splatting.base_model import GaussianModel
from models.splatting.hierarchical.hierarchy_utils import (aggregate_gaussians_recursively,
                                                           assign_unique_values, build_octree,
                                                           calculate_weights)
from utils.general_utils import compute_required_rotation_and_scaling, pack_full_to_lower_triangular


class MultiRateMeanScaleHyperprior(CompressionModel):
    def __init__(
        self,
        dataset: Namespace,
        M: int,
        N: int | None = None,
        n_levels: int = 1,
        scale_mag: int = 20,
        extra_pos_scale: bool = False,
        with_positions: bool = False,
    ):
        """Mean-Scale Hyperprior compressor.

        Args:
            dataset (Namespace): The dataset namespace.
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
            n_levels (int, optional): Number of levels in the entropy model. Defaults to 1.
            scale_mag (int, optional): Initial scale magnitude. Defaults to 400.
            extra_pos_scale (bool, optional): Whether to use extra scaling for the first 6
                coefficients (position and scaling). Defaults to False.
        """
        super(MultiRateMeanScaleHyperprior, self).__init__()

        self.h_a = nn.Sequential(
            nn.Linear(M, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N),
        )

        self.M = M
        self.N = N

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        # self.normalizer = nn.BatchNorm1d(M)

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

        self.scale = nn.Parameter(
            scale_mag * torch.ones((n_levels, M), requires_grad=requires_grad),
            requires_grad=requires_grad,
        )
        self.hyper_scale = nn.Parameter(
            scale_mag * torch.ones((n_levels, N), requires_grad=requires_grad),
            requires_grad=requires_grad,
        )

        with torch.no_grad():
            if extra_pos_scale and with_positions:
                self.scale[:, 3:9] = 10 * self.scale[:, 3:9]
            elif extra_pos_scale:
                self.scale[:, :6] = 10 * self.scale[:, :6]

    def capture(self) -> Dict:
        """Capture the state of the model."""
        return self.state_dict()

    def restore(self, state: Dict):
        """Restore the state of the model."""
        message = self.load_state_dict(state)
        return message

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
            scale = (torch.abs(self.scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.scale[int(level) + 1]) ** interp
            )
        else:
            scale = torch.abs(self.scale[int(level)])

        attrs_to_compress = attrs_to_compress * scale.unsqueeze(0)
        return attrs_to_compress

    def descale_inputs(self, decoded_attrs, level: int):
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
            descale = (torch.abs(self.scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.scale[int(level) + 1]) ** interp
            )
        else:
            descale = torch.abs(self.scale[int(level)])

        decoded_attrs = decoded_attrs / descale.unsqueeze(0)
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

    def forward(self, attrs: torch.Tensor, level: int | float):
        """Forward pass through the mean-scale hyperprior compressor for training

        Args:
            attrs (torch.Tensor): The attributes to compress.
            level (int | float): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        # sampled_attributes = (sampled_attributes - self.mean_vals) / self.std_vals

        # normalized_attrs = self.normalizer(attrs)
        normalized_attrs = attrs

        attrs_to_compress = self.scale_inputs(normalized_attrs, level)
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

        return (
            decoded_attrs,
            {"y": y_likelihoods, "z": z_likelihoods},
            num_coeffs,
        )

    def compress(
        self, attrs: torch.Tensor, level: int | float
    ) -> Dict[str, List[bytearray] | List[int]]:
        """Actual compression method used for inference time.

        Args:
            attrs (torch.Tensor): The attributes to compress.
            level (int | float): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        # attributes = (attributes - self.mean_vals) / self.std_vals
        # normalized_attrs = self.normalizer(attrs)
        normalized_attrs = attrs

        attrs_to_compress = self.scale_inputs(normalized_attrs, level)
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
            "shapes": [attrs_to_compress.shape[0]],
            "sizes": [
                (len(string_y) + len(string_z)) / (2**20)
                for string_y, string_z in zip(y_strings, z_strings)
            ],
            "level": level,
        }

    def decompress(
        self, strings: List[bytearray], size: List[int], level: int | float
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

        return decoded_attrs


class CompleteMeanScaleHyperprior(nn.Module):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """Complete mean-scale hyperprior compressor.
        - One entropy bottleneck for intra-frame compression.
        - One entropy bottleneck for inter-frame compression.

        Args:
            dataset (Namespace): The dataset namespace.
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
            n_levels (int, optional): Number of levels in the entropy model. Defaults to 1.
        """
        super(CompleteMeanScaleHyperprior, self).__init__()

        self.dataset = dataset
        self.n_quant_levels = n_levels

        intra_M = (3 * ((dataset.sh_degree + 1) ** 2)) + 0 + 6 + 1
        inter_M = (3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 6 + 1

        self.intra_compressor = MultiRateMeanScaleHyperprior(
            dataset, intra_M, N, n_levels, scale_mag=100, extra_pos_scale=True, with_positions=False
        )
        self.inter_compressor = MultiRateMeanScaleHyperprior(
            dataset, inter_M, N, n_levels, scale_mag=100, extra_pos_scale=True, with_positions=True
        )

    def training_setup(self, optimization_args: Namespace):
        """Set up the optimizer for the complete entropy bottleneck compressor."""

        l = [
            {
                "params": [
                    p
                    for n, p in self.intra_compressor.named_parameters()
                    if (not n.endswith(".quantiles")) and ("scale" not in n)
                ],
                "lr": optimization_args.compressor_lr,
                "name": "intra_compressor_params",
            },
            {
                "params": [
                    p
                    for n, p in self.inter_compressor.named_parameters()
                    if (not n.endswith(".quantiles")) and ("scale" not in n)
                ],
                "lr": optimization_args.compressor_lr,
                "name": "inter_compressor_params",
            },
            {
                "params": [p for n, p in self.intra_compressor.named_parameters() if "scale" in n]
                + [p for n, p in self.inter_compressor.named_parameters() if "scale" in n],
                "lr": 0.1,
                "name": "scale_params",
            },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        l_aux = [
            {
                "params": [
                    p
                    for n, p in self.intra_compressor.entropy_bottleneck.named_parameters()
                    if n.endswith(".quantiles")
                ],
                "lr": optimization_args.compressor_aux_lr,
                "name": "intra_aux_params",
            },
            {
                "params": [
                    p
                    for n, p in self.inter_compressor.entropy_bottleneck.named_parameters()
                    if n.endswith(".quantiles")
                ],
                "lr": optimization_args.compressor_aux_lr,
                "name": "inter_aux_params",
            },
        ]
        self.aux_optimizer = torch.optim.Adam(l_aux, lr=0.0, eps=1e-15)

    def capture(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """Capture the state of the model amd the optimizers"""
        return (
            self.intra_compressor.capture(),
            self.inter_compressor.capture(),
            self.optimizer.state_dict(),
            self.aux_optimizer.state_dict(),
        )

    def capture_model_params(
        self, intra_num_gaussians: int, inter_num_gaussians: int, active_sh_degree: int
    ) -> Tuple[Dict, Dict]:
        """Capture the model parameters for test time. This capturing contains
        only vital model parameters.

        Args:
            intra_num_gaussians (int): The number of intra-frame Gaussians (required for decode)
            inter_num_gaussians (int): The number of inter-frame Gaussians (required for decode)
            sh_degree (int): The spherical harmonics degree.
        """
        return (
            self.intra_compressor.capture(),
            self.inter_compressor.capture(),
            intra_num_gaussians,
            inter_num_gaussians,
            active_sh_degree,
        )

    def restore(self, state: Tuple[Dict, Dict, Dict, Dict]):
        """Restore the state of the model and the optimizers.
        This is used for resuming training.
        """
        msg1 = self.intra_compressor.restore(state[0])
        msg2 = self.inter_compressor.restore(state[1])
        self.optimizer.load_state_dict(state[2])
        self.aux_optimizer.load_state_dict(state[3])

    def restore_model_params(self, model_params: Tuple[Dict, Dict]):
        """Restore the model parameters for test time. This restoring contains
        only vital model parameters.
        """
        msg1 = self.intra_compressor.restore(model_params[0])
        msg2 = self.inter_compressor.restore(model_params[1])

    def update(self):
        """Update the entropy bottleneck models for the quantiles and inference time compression."""
        self.intra_compressor.update(update_quantiles=True)
        self.inter_compressor.update(update_quantiles=True)

    def aux_loss(self):
        """Calculate the auxiliary loss for the entropy bottleneck models."""
        return self.intra_compressor.aux_loss() + self.inter_compressor.aux_loss()

    @classmethod
    def calculate_entropy(self, y_likelihoods: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the entropy from the likelihoods of the entropy bottleneck."""
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in y_likelihoods.values()
        )

    # def get_cut_attributes(
    #     self,
    #     gaussians: GaussianModel,
    #     chosen_depth: int,
    #     max_depth: int,
    # ):
    #     import build_octree as hierarchy_generation
    #     assert chosen_depth <= max_depth, "Chosen depth must be less than or equal to max depth"

    #     if chosen_depth == -1:
    #         return (
    #             gaussians.get_xyz,
    #             gaussians.get_features,
    #             gaussians.get_covariance(),
    #             gaussians.get_opacity,
    #             gaussians.active_sh_degree,
    #         )

    #     num_cubes = 2 ** chosen_depth
    #     # node_assignments = build_octree(gaussians.get_xyz, num_cubes)

    #     with torch.no_grad():
    #         point3D = gaussians.get_xyz.contiguous()
    #         aabb_min = torch.min(point3D, dim=0)[0]
    #         aabb_max = torch.max(point3D, dim=0)[0]

    #         box_d = aabb_max - aabb_min
    #         box_min = (aabb_min - 0.01 * box_d).contiguous()
    #         box_max = (aabb_max + 0.01 * box_d).contiguous()

    #         cube_size = torch.max(box_max - box_min) / num_cubes

    #         # Compute the grid indices for each point
    #         indices = ((point3D - box_min) / cube_size).floor().long()

    #         # Convert 3D indices to a unique group ID
    #         # Assuming the grid extends only within the AABB:
    #         grid_dim = ((box_max - box_min) / cube_size).ceil().long()
    #         group_ids = indices[:, 0] + indices[:, 1] * grid_dim[0] + indices[:, 2] * grid_dim[0] * grid_dim[1]

    #         _, node_assignments = torch.unique(group_ids, sorted=True, return_inverse=True)

    #     mu = gaussians.get_xyz.contiguous()
    #     scaling = gaussians.get_scaling.contiguous()
    #     rotation = gaussians.get_rotation.contiguous()
    #     sigma = gaussians.get_covariance().contiguous()
    #     opacity = gaussians.get_opacity.contiguous()
    #     sh = gaussians.get_features.contiguous()

    #     weights = calculate_weights(sigma, opacity).contiguous()

    #     # # Call function to perform aggregation
    #     # node_xyzs, node_sigma, node_opacity, node_shs = AggregationFunction.apply(
    #     #     weights,
    #     #     mu,
    #     #     sigma,
    #     #     opacity,
    #     #     sh.view(sh.shape[0], -1),
    #     #     node_assignments.type(torch.int32),
    #     # )

    #     # Number of groups
    #     num_nodes = torch.max(node_assignments).item() + 1

    #     # Aggregate weights
    #     node_weights = torch.zeros(num_nodes, 1).cuda()
    #     node_weights.scatter_add_(0, node_assignments.unsqueeze(1), weights)

    #     # Aggregate means
    #     _gaussian_means = weights * mu
    #     node_xyzs = torch.zeros(num_nodes, _gaussian_means.shape[1]).cuda()
    #     node_xyzs = node_xyzs.scatter_add(
    #         0,
    #         node_assignments.unsqueeze(1).expand(-1, _gaussian_means.shape[1]),
    #         _gaussian_means
    #     ) / node_weights

    #     # Aggregate scaling
    #     _gaussian_scaling = weights * scaling
    #     node_scaling = torch.zeros(num_nodes, _gaussian_scaling.shape[1]).cuda()
    #     node_scaling = node_scaling.scatter_add(
    #         0,
    #         node_assignments.unsqueeze(1).expand(-1, _gaussian_scaling.shape[1]),
    #         _gaussian_scaling
    #     ) / node_weights

    #     # Aggregate rotation
    #     _gaussian_rotation = weights * rotation
    #     node_rotation = torch.zeros(num_nodes, _gaussian_rotation.shape[1]).cuda()
    #     node_rotation = node_rotation.scatter_add(
    #         0,
    #         node_assignments.unsqueeze(1).expand(-1, _gaussian_rotation.shape[1]),
    #         _gaussian_rotation
    #     ) / node_weights

    #     # Aggregate sigma
    #     _gaussian_sigma = weights * sigma
    #     node_sigma = torch.zeros(num_nodes, _gaussian_sigma.shape[1]).cuda()

    #     diff_mu = mu - node_xyzs[node_assignments]
    #     diff_mu_outer = torch.einsum("ij,ik->ijk", diff_mu, diff_mu)
    #     diff_mu_vector = pack_full_to_lower_triangular(diff_mu_outer)
    #     _gaussian_sigma = _gaussian_sigma + weights * diff_mu_vector

    #     node_sigma = torch.zeros(num_nodes, _gaussian_sigma.shape[1]).cuda()
    #     node_sigma = node_sigma.scatter_add(
    #         0,
    #         node_assignments.unsqueeze(1).expand(-1, _gaussian_sigma.shape[1]),
    #         _gaussian_sigma
    #     ) / node_weights

    #     # Aggregate opacity
    #     _gaussian_opacity = weights * opacity
    #     node_opacity = torch.zeros(num_nodes, 1).cuda()
    #     node_opacity = node_opacity.scatter_add(
    #         0,
    #         node_assignments.unsqueeze(1).expand(-1, _gaussian_opacity.shape[1]),
    #         _gaussian_opacity
    #     ) / node_weights

    #     # Aggregate SH
    #     _gaussian_shs = weights * sh.view(sh.shape[0], -1)
    #     node_shs = torch.zeros(num_nodes, _gaussian_shs.shape[1]).cuda()
    #     node_shs = node_shs.scatter_add(
    #         0,
    #         node_assignments.unsqueeze(1).expand(-1, _gaussian_shs.shape[1]),
    #         _gaussian_shs
    #     ) / node_weights

    #     return (
    #         node_xyzs,
    #         node_sigma,
    #         node_shs.view(node_shs.shape[0], -1, 3),
    #         node_opacity,
    #         node_assignments.type(torch.int32)
    #     )

    def get_cut_attributes(
        self,
        gaussians: GaussianModel,
        chosen_depth: int,
        max_depth: int,
    ):
        """Get the cut attributes for the given depth level. This method retrieves the intermediate
        node features for the Gaussian primitives. These features are
        - xyz: The position of the node.
        - features: The SH coefficients of the node.
        - covariance: The covariance of the node.
        - opacity: The opacity of the node.

        Args:
            gaussians (GaussianModel): The Gaussian model.
            chosen_depth (int): The depth level to cut the octree.
            max_depth (int): The maximum depth of the octree.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The
                cut attributes and node assignment of Gaussian primitives.
        """
        assert chosen_depth <= max_depth, "Chosen depth must be less than or equal to max depth"

        # If the depth is -1, we return the Gaussian primitives
        if chosen_depth == -1:
            return (
                gaussians.get_xyz,
                gaussians.get_features,
                gaussians.get_covariance(),
                gaussians.get_opacity,
                gaussians.active_sh_degree,
            )

        # Build the octree and get the node assignments
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

        # Aggregate the Gaussian primitives to the chosen depth level
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

    def get_subsample_indices(self, subsample_fraction: float, num_gaussians: int):
        """Get the subsample indices for the Gaussian primitives.

        Args:
            subsample_fraction (float): The fraction of Gaussian primitives to sample.
            num_gaussians (int): The total number of Gaussian primitives.
        """
        if subsample_fraction < 1.0 - 1e-6:
            total_gaussians = num_gaussians
            sample_size = int(total_gaussians * subsample_fraction)
            indices = torch.randperm(total_gaussians)[:sample_size]
        else:
            indices = torch.arange(num_gaussians)

        return indices

    def get_residuals(
        self,
        position: torch.Tensor,
        covariance: torch.Tensor,
        features: torch.Tensor,
        opacity: torch.Tensor,
        agg_xyz: torch.Tensor,
        agg_covariance: torch.Tensor,
        agg_features: torch.Tensor,
        agg_opacities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the residuals for the Gaussian primitives.
        The agg_* parameters are intermediate node features of the Gaussian primitives used
        for the `prediction`. The gaussians are the original Gaussian primitives. The residuals
        are calculated as the difference between the original Gaussian primitives and the
        aggregated node features.
        """

        residual_position = position - agg_xyz
        # residual_scaling, residual_rotation = compute_required_rotation_and_scaling(
        #     agg_covariance, covariance
        # )
        residual_covariance = covariance - agg_covariance
        residual_features = features - agg_features
        residual_opacities = opacity - agg_opacities
        return (
            residual_position,
            # residual_scaling,
            # residual_rotation,
            residual_covariance,
            residual_features,
            residual_opacities,
        )

    def forward_intra_params(self, attrs: torch.Tensor, level: int):
        """Forward pass through the intra-frame entropy bottleneck for training.

        Args:
            attrs (torch.Tensor): The attributes to compress.
            level (int): The quantization level to use.
            subsample_fraction (float, optional): The fraction of Gaussian primitives to sample.
                Defaults to 0.1. (not needed)
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        compressed_attrs, likelihoods, num_coeffs = self.intra_compressor(attrs, level)
        return compressed_attrs, likelihoods, num_coeffs

    def forward_inter_params(self, attrs: torch.Tensor, level: int):
        """Forward pass through the inter-frame entropy bottleneck for training.

        Args:
            attrs (torch.Tensor): The attributes to compress.
            level (int): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        compressed_attrs, likelihoods, num_coeffs = self.inter_compressor(attrs, level)
        return compressed_attrs, likelihoods, num_coeffs

    def compress_intra(self, attrs: torch.Tensor, level: int) -> Dict[str, List[bytearray]]:
        """Actual compression method used for inference time for intra-frame compression.

        Args:
            attrs (torch.Tensor): The attributes to compress.
            level (int): The quantization level to use.
        """

        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        return self.intra_compressor.compress(attrs, level)

    def decompress_intra(
        self, string: List[bytearray], size: List[int], level: int
    ) -> Dict[str, torch.Tensor]:
        """Actual decompression method used for inference time for intra-frame compression.

        Args:
            string (List[bytearray]): The compressed bitstrings that will be used for decoding.
            size (List[int]): The size of the compressed bitstrings (number of Gaussians).
            level (int): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        return self.intra_compressor.decompress(string, size, level)

    def compress_inter(self, attrs: torch.Tensor, level: int) -> Dict[str, List[bytearray]]:
        """Actual compression method used for inference time for inter-frame compression.

        Args:
            attrs (torch.Tensor): The attributes to compress.
            level (int): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        return self.inter_compressor.compress(attrs, level)

    def decompress_inter(
        self, string: List[bytearray], size: List[int], level: int
    ) -> Dict[str, torch.Tensor]:
        """Actual decompression method used for inference time for inter-frame compression.

        Args:
            string (List[bytearray]): The compressed bitstrings that will be used for decoding.
            size (List[int]): The size of the compressed bitstrings (number of Gaussians).
            level (int): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        return self.inter_compressor.decompress(string, size, level)
