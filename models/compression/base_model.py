import math
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from compressai.models import CompressionModel
from torch import Tensor
from torch.optim.optimizer import Optimizer


class BaseCompressor(CompressionModel, ABC):

    def __init__(self, dataset, M: int | None = None, N: int | None = None, levels: int = 4):
        """Base compressor class.

        Args:
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
            levels (int, optional): Number of levels in the entropy model. Defaults to 4.
        """
        super(CompressionModel, self).__init__()

        self.optimizer: Optimizer | None = None
        self.aux_optimizer: Optimizer | None = None

        if dataset.compression_fixed_quantization:
            requires_grad = False
        else:
            requires_grad = True

        self.scaling_scale = nn.Parameter(
            20 * torch.ones(1, requires_grad=requires_grad), requires_grad=requires_grad
        )
        self.rotation_scale = nn.Parameter(
            20 * torch.ones(1, requires_grad=requires_grad), requires_grad=requires_grad
        )
        self.opacity_scale = nn.Parameter(
            20 * torch.ones(1, requires_grad=requires_grad), requires_grad=requires_grad
        )
        self.sh_scale = nn.Parameter(
            20 * torch.ones(((dataset.sh_degree + 1) ** 2), requires_grad=requires_grad),
            requires_grad=requires_grad,
        )

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        """Load the state dictionary of the compressor.

        Args:
            state_dict (Dict): The state dictionary to load.
            strict (bool, optional): Whether to strictly enforce that the keys
                in the state dictionary match the keys in the model. Defaults to True.
        """
        message = super(BaseCompressor, self).load_state_dict(state_dict, strict)
        return message

    def capture(self) -> Tuple[Dict, Tensor, Tensor, Tensor, Tensor, Dict, Dict]:
        """Capture the state dictionary of the compressor."""
        return (
            self.state_dict(),
            self.scaling_scale,
            self.rotation_scale,
            self.opacity_scale,
            self.sh_scale,
            self.optimizer.state_dict(),
            self.aux_optimizer.state_dict() if self.aux_optimizer is not None else None,
        )

    def capture_compressor_params(
        self, num_gaussians: int, active_sh_degree: int
    ) -> Tuple[Dict, Tensor, Tensor, Tensor, Tensor, int, int]:
        """Capture the compressor parameters for compression of the vital parameters.
        This method is used for testing the compression.
        """
        return (
            self.state_dict(),
            self.scaling_scale,
            self.rotation_scale,
            self.opacity_scale,
            self.sh_scale,
            num_gaussians,
            active_sh_degree,
        )

    def restore_compressor_params(
        self,
        compressor_params: Tuple[
            Dict,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
        ],
    ):
        """Restore the compressor parameters for compression of the vital parameters.
        This method is used for testing the compression.
        """
        message = self.load_state_dict(compressor_params[0])
        self.scaling_scale = compressor_params[1]
        self.rotation_scale = compressor_params[2]
        self.opacity_scale = compressor_params[3]
        self.sh_scale = compressor_params[4]
        return message

    def restore_model(
        self,
        model_args: (
            Tuple[
                Dict,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Dict,
                Dict,
            ]
            | Dict
        ),
    ):
        """Restore the model from the captured tuple."""
        message = self.load_state_dict(model_args[0])
        self.scaling_scale = model_args[1]
        self.rotation_scale = model_args[2]
        self.opacity_scale = model_args[3]
        self.sh_scale = model_args[4]
        return message

    def restore_optimizer(self, optimizer_args: Dict, aux_optimizer_args: Dict):
        self.optimizer.load_state_dict(optimizer_args)

        if aux_optimizer_args is not None:
            self.aux_optimizer.load_state_dict(aux_optimizer_args)

    def optimize(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.aux_optimizer is not None:
            self.aux_optimizer.step()
            self.aux_optimizer.zero_grad()

    def init_normalization_params(self, gaussians):
        spherical_harmonics_reshaped = gaussians.get_features.flatten(1, 2)

        attrs_to_compress = torch.cat(
            (
                spherical_harmonics_reshaped,
                gaussians._scaling,
                gaussians._rotation,
                gaussians._opacity,
            ),
            dim=1,
        )

        # Normalize the attributes using mean and std
        self.mean_vals = attrs_to_compress.mean(0, keepdim=True).requires_grad_(False)
        self.std_vals = attrs_to_compress.std(0, keepdim=True).requires_grad_(False)

    def normalize_concat_inputs(
        self,
        sampled_features: torch.Tensor,
        sampled_scales: torch.Tensor,
        sampled_rotations: torch.Tensor,
        sampled_opacity: torch.Tensor,
    ):
        # spherical_harmonics_reshaped: [N, C] shape
        spherical_harmonics_reshaped = sampled_features.flatten(1, 2)

        # rotations_scales_reshaped: [N, 7] shape
        rotations_scales_reshaped = torch.cat((sampled_scales, sampled_rotations), dim=1)

        attrs_to_compress = (
            torch.cat(
                (spherical_harmonics_reshaped, rotations_scales_reshaped, sampled_opacity), dim=1
            )
            - self.mean_vals
        ) / self.std_vals
        # attrs_to_compress = spherical_harmonics_reshaped

        attrs_to_compress = attrs_to_compress * torch.cat(
            (
                self.sh_scale.repeat_interleave(3),
                self.scaling_scale.repeat_interleave(3),
                self.rotation_scale.repeat_interleave(4),
                self.opacity_scale,
            ),
            dim=0,
        ).unsqueeze(0)

        return attrs_to_compress

    def denormalize_split_inputs(self, decoded_attrs, active_sh_degree):
        # # Denormalize the decoded attributes
        # decoded_attrs = (decoded_attrs / 100.) * self.std_vals.unsqueeze(-1) + self.mean_vals.unsqueeze(-1)
        decoded_attrs = torch.permute(decoded_attrs[0], (1, 0)) / (
            torch.cat(
                (
                    self.sh_scale.repeat_interleave(3),
                    self.scaling_scale.repeat_interleave(3),
                    self.rotation_scale.repeat_interleave(4),
                    self.opacity_scale,
                ),
                dim=0,
            ).unsqueeze(0)
        )

        decoded_attrs = decoded_attrs * self.std_vals + self.mean_vals

        # Split the compressed attributes
        compressed_harmonics = decoded_attrs[:, : 3 * ((active_sh_degree + 1) ** 2)]
        compressed_scales_rotations = decoded_attrs[:, 3 * ((active_sh_degree + 1) ** 2) : -1]

        compressed_features = compressed_harmonics.view(-1, (active_sh_degree + 1) ** 2, 3)
        compressed_scales = compressed_scales_rotations[:, :3]
        compressed_rotations = compressed_scales_rotations[:, 3:]

        compressed_opacity = decoded_attrs[:, -1:]

        return compressed_features, compressed_scales, compressed_rotations, compressed_opacity

    def sample_gaussians(self, gaussians, fraction=0.1):
        """Sample a fraction of the gaussians for training the compressor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                sampled_features, sampled_opacity, sampled_scales, sampled_rotations, indices"""

        if fraction < 1.0 - 1e-6:
            total_gaussians = len(gaussians.get_xyz)
            sample_size = int(total_gaussians * fraction)
            indices = torch.randperm(total_gaussians)[:sample_size]
        else:
            indices = torch.arange(len(gaussians.get_xyz))

        sampled_features = gaussians.get_features[indices]
        sampled_opacity = gaussians._opacity[indices]
        sampled_scales = gaussians._scaling[indices]
        sampled_rotations = gaussians._rotation[indices]

        return (
            sampled_features,
            sampled_opacity,
            sampled_scales,
            sampled_rotations,
            indices,
        )

    @abstractmethod
    def training_setup(self, optimization_args: Namespace):
        pass

    @abstractmethod
    def forward(self, y: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def compress(self, x: torch.Tensor) -> List[bytearray]:
        pass

    @abstractmethod
    def decompress(self, string: List[str], size: List[int], active_sh_degree: int) -> torch.Tensor:
        pass

    # @abstractmethod
    # def get_bitstring(
    #     self, gaussians: BaseModel
    # ) -> Dict[str, List[bytearray] | List[float]]:
    #     pass

    # @abstractmethod
    # def get_compressed_spherical_harmonics(
    #     self, string: List[str], num_gaussians: int
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     pass

    @classmethod
    def calculate_entropy(self, y_likelihoods: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in y_likelihoods.values()
        )
