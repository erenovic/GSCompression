import math
from argparse import Namespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from compressai import entropy_models
from compressai.models import CompressionModel

from models.splatting.base_model import GaussianModel
from models.splatting.hierarchical.hierarchy_utils import (aggregate_gaussians_recursively,
                                                           assign_unique_values, build_octree,
                                                           calculate_weights)


class MultiRateEntropyBottleneck(CompressionModel):
    def __init__(
        self,
        dataset: Namespace,
        M: int,
        N: int | None = None,
        n_levels: int = 1,
        scale_mag: int = 400,
        extra_pos_scale: bool = False,
    ):
        """Entropy bottleneck compressor.

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
        super(MultiRateEntropyBottleneck, self).__init__()

        self.entropy_bottleneck = entropy_models.EntropyBottleneck(M)

        if dataset.compression_fixed_quantization:
            print("Fixed quantization")
            requires_grad = False
        else:
            print("Learnable quantization")
            requires_grad = True

        self.n_quant_levels = n_levels

        self.scale = nn.Parameter(
            scale_mag * torch.ones((n_levels, M), requires_grad=requires_grad),
            requires_grad=requires_grad,
        )
        if extra_pos_scale:
            with torch.no_grad():
                self.scale[:, :6] = 10 * self.scale[:, :6]

    def capture(self) -> Dict:
        """Capture the state of the model."""
        return self.state_dict()

    def restore(self, state: Dict):
        """Restore the state of the model."""
        self.load_state_dict(state)

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
                "params": [self.scale],
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
            scale = (torch.abs(self.scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.scale[int(level) + 1]) ** interp
            )
        else:
            scale = torch.abs(self.scale[int(level)])

        attrs_to_compress = attrs_to_compress * scale.unsqueeze(0)
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
            descale = (torch.abs(self.scale[int(level)]) ** (1 - interp)) * (
                torch.abs(self.scale[int(level) + 1]) ** interp
            )
        else:
            descale = torch.abs(self.scale[int(level)])

        decoded_attrs = decoded_attrs / descale.unsqueeze(0)
        return decoded_attrs

    def forward(self, attrs: torch.Tensor, level: int | float):
        """Forward pass through the entropy bottleneck for training.

        Args:
            attrs (torch.Tensor): The attributes to compress.
            level (int | float): The quantization level to use.
        """
        assert (
            level < self.n_quant_levels
        ), f"Level {level} is greater than the number of quantization levels {self.n_quant_levels}"

        attrs_to_compress = self.scale_inputs(attrs, level)
        num_coeffs = attrs_to_compress.shape[1]

        # attrs_to_compress = spherical_harmonics_reshaped
        attrs_to_compress = attrs_to_compress.permute(1, 0).unsqueeze(0)

        z_hat, z_likelihoods = self.entropy_bottleneck(attrs_to_compress)

        decoded_attrs = self.descale_inputs(z_hat.squeeze(0).permute(1, 0), level)

        return (decoded_attrs, {"z": z_likelihoods}, num_coeffs)

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
        attrs_to_compress = self.scale_inputs(attrs, level)
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
        self, string: List[bytearray], size: List[int], level: int | float
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

        decompressed_attrs = self.entropy_bottleneck.decompress(string[0], size)

        decoded_attrs = self.descale_inputs(decompressed_attrs.squeeze(0).permute(1, 0), level)

        return decoded_attrs


class CompleteEntropyBottleneck(nn.Module):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """Complete entropy bottleneck compressor.
        - One entropy bottleneck for intra-frame compression.
        - One entropy bottleneck for inter-frame compression.

        Args:
            dataset (Namespace): The dataset namespace.
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
            n_levels (int, optional): Number of levels in the entropy model. Defaults to 1.
        """
        super(CompleteEntropyBottleneck, self).__init__()

        self.dataset = dataset
        self.n_quant_levels = n_levels

        intra_M = (3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 7 + 1
        inter_M = (3 * ((dataset.sh_degree + 1) ** 2)) + 3 + 7 + 1

        self.intra_compressor = MultiRateEntropyBottleneck(
            dataset, intra_M, N, n_levels, scale_mag=20
        )
        self.inter_compressor = MultiRateEntropyBottleneck(
            dataset, inter_M, N, n_levels, scale_mag=20, extra_pos_scale=True
        )

    def training_setup(self, optimization_args: Namespace):
        """Set up the optimizer for the complete entropy bottleneck compressor."""

        l = [
            {
                "params": [
                    p
                    for n, p in self.intra_compressor.entropy_bottleneck.named_parameters()
                    if not n.endswith(".quantiles")
                ],
                "lr": optimization_args.compressor_lr,
                "name": "intra_compressor_params",
            },
            {
                "params": [
                    p
                    for n, p in self.inter_compressor.entropy_bottleneck.named_parameters()
                    if not n.endswith(".quantiles")
                ],
                "lr": optimization_args.compressor_lr,
                "name": "inter_compressor_params",
            },
            {
                "params": [self.intra_compressor.scale],
                "lr": optimization_args.compressor_lr,
                "name": "intra_scaling_params",
            },
            {
                "params": [self.inter_compressor.scale],
                "lr": optimization_args.compressor_lr,
                "name": "inter_scaling_params",
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
        self, intra_num_gaussians: int, inter_num_gaussians: int, sh_degree: int
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
            sh_degree,
        )

    def restore(self, state: Tuple[Dict, Dict, Dict, Dict]):
        """Restore the state of the model and the optimizers.
        This is used for resuming training.
        """
        self.intra_compressor.restore(state[0])
        self.inter_compressor.restore(state[1])
        self.optimizer.load_state_dict(state[2])
        self.aux_optimizer.load_state_dict(state[3])

    def restore_model_params(self, model_params: Tuple[Dict, Dict]):
        """Restore the model parameters for test time. This restoring contains
        only vital model parameters.
        """
        self.intra_compressor.restore(model_params[0])
        self.inter_compressor.restore(model_params[1])

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
        gaussians: GaussianModel,
        agg_xyz: torch.Tensor,
        agg_scaling: torch.Tensor,
        agg_rotation: torch.Tensor,
        agg_features: torch.Tensor,
        agg_opacities: torch.Tensor,
        sorted_node_assignments: torch.Tensor,
        sorted_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the residuals for the Gaussian primitives.
        The agg_* parameters are intermediate node features of the Gaussian primitives used
        for the `prediction`. The gaussians are the original Gaussian primitives. The residuals
        are calculated as the difference between the original Gaussian primitives and the
        aggregated node features.
        """

        residual_position = gaussians.get_xyz[sorted_indices] - agg_xyz[sorted_node_assignments]
        residual_scaling = (
            gaussians.get_scaling[sorted_indices] - agg_scaling[sorted_node_assignments]
        )
        residual_rotation = (
            gaussians.get_rotation[sorted_indices] - agg_rotation[sorted_node_assignments]
        )
        residual_features = (
            gaussians.get_features[sorted_indices] - agg_features[sorted_node_assignments]
        )
        residual_opacities = (
            gaussians.get_opacity[sorted_indices] - agg_opacities[sorted_node_assignments]
        )
        return (
            residual_position,
            residual_scaling,
            residual_rotation,
            residual_features,
            residual_opacities,
        )

    def forward_intra_params(
        self, attrs: torch.Tensor, level: int, subsample_fraction: float = 0.1
    ):
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
