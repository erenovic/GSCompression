import math
from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel
import logging

from models.compression.base_model import BaseCompressor
from models.splatting.hierarchical.hierarchy_utils import (
    aggregate_gaussians_recursively,
    assign_unique_values,
    build_octree,
    calculate_weights,
    AggregationFunction
)
from models.splatting.mcmc_model import GaussianModel
from utils.general_utils import (
    build_covariance_from_scaling_rotation,
    build_rotation,
    conjugate_quaternion,
    pack_full_to_lower_triangular,
    unpack_lower_triangular_to_full,
)


class CodecNet(CompressionModel):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """CodecNet.

        Inspired from:
        https://github.com/Orange-OpenSource/AIVC/tree/3ce96717ea749d70b308a4ec360f1bb943405c14

        Args:
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
        """
        super(CodecNet, self).__init__()

        self.n_levels = n_levels

        M = 3 * ((dataset.sh_degree + 1) ** 2) + 3 + 3 + 4 + 1
        self.M = M

        self.g_a = nn.Sequential(
            nn.Linear(2 * M, 3 * M // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3 * M // 2, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, N),
        )

        self.g_s = nn.Sequential(
            nn.Linear(N, 3 * N // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3 * N // 2, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, M),
        )

        self.h_a = nn.Sequential(
            nn.Linear(N, N // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N // 2, N // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N // 2, N // 2),
        )

        self.h_s = nn.Sequential(
            nn.Linear(N // 2, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N * 3 // 2, 2 * N),
        )

        self.entropy_bottleneck = EntropyBottleneck(N // 2)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_scales = nn.Parameter(
            20 * torch.ones((n_levels, N), requires_grad=True), requires_grad=True
        )
        self.h_scales = nn.Parameter(
            20 * torch.ones((n_levels, N // 2), requires_grad=True), requires_grad=True
        )

    def scale_latents(self, y: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            g_scale = (torch.abs(self.g_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.g_scales[int(level) + 1]) ** interp
            )

        else:
            g_scale = torch.abs(self.g_scales[int(level)])

        return y * g_scale.unsqueeze(0)

    def descale_latents(self, y_hat: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            g_scale = (torch.abs(self.g_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.g_scales[int(level) + 1]) ** interp
            )
        else:
            g_scale = torch.abs(self.g_scales[int(level)])

        return y_hat / g_scale.unsqueeze(0)

    def scale_hyper_latents(self, z: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            h_scale = (torch.abs(self.h_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.h_scales[int(level) + 1]) ** interp
            )

        else:
            h_scale = torch.abs(self.h_scales[int(level)])

        return z * h_scale.unsqueeze(0)

    def descale_hyper_latents(self, z_hat: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            h_scale = (torch.abs(self.h_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.h_scales[int(level) + 1]) ** interp
            )
        else:
            h_scale = torch.abs(self.h_scales[int(level)])

        return z_hat / h_scale.unsqueeze(0)

    def forward(
        self, x_pred: torch.Tensor, x: torch.Tensor, level: int, training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """CodecNet forward pass.

        Args:
            x (torch.Tensor): Actual Gaussians [N, C], multiplied by \alpha.
            x_pred (torch.Tensor): Predicted tensor [N, C], multiplied by \alpha.

        Returns:
            y: Compressed and corrective Gaussians [N, C].
        """
        x_cat = torch.cat((x_pred, x), dim=1)

        y = self.g_a(x_cat)
        y_scaled = self.scale_latents(y, level)

        z = self.h_a(y_scaled)
        z_scaled = self.scale_hyper_latents(z, level)
        z_permute = torch.permute(z_scaled, (1, 0)).unsqueeze(0)  # [1, C, N]

        z_hat, z_likelihoods = self.entropy_bottleneck(z_permute, training=training)

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)  # [N, C]
        z_descaled = self.descale_hyper_latents(z_hat_permute, level)

        entropy_params = self.h_s(z_descaled)
        scales_hat, means_hat = entropy_params.chunk(2, 1)

        scales_hat_permute = torch.permute(scales_hat, (1, 0)).unsqueeze(0)
        means_hat_permute = torch.permute(means_hat, (1, 0)).unsqueeze(0)

        y_permute = y_scaled.permute(1, 0).unsqueeze(0)  # [1, C, N]

        compressed_y, y_likelihoods = self.gaussian_conditional(
            y_permute, scales_hat_permute, means=means_hat_permute, training=training
        )
        compressed_y_permute = compressed_y.squeeze(0).permute(1, 0)  # [N, C]
        y_descaled = self.descale_latents(compressed_y_permute, level)

        output = self.g_s(y_descaled)

        return output, {"y": y_likelihoods, "z": z_likelihoods}


class MSGNet(CompressionModel):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """MSGNet.

        Inspired from:
        https://github.com/Orange-OpenSource/AIVC/tree/3ce96717ea749d70b308a4ec360f1bb943405c14

        Args:
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
        """
        super(MSGNet, self).__init__()

        self.n_levels = n_levels

        M = 3 * ((dataset.sh_degree + 1) ** 2) + 3 + 3 + 4 + 1
        self.M = M

        self.g_a = nn.Sequential(
            nn.Linear(2 * M, 3 * M // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3 * M // 2, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, N),
        )

        self.g_s = nn.Sequential(
            nn.Linear(N, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, 3 * M // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3 * M // 2, 2 * M),
        )

        self.h_a = nn.Sequential(
            nn.Linear(N, N // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N // 2, N // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N // 2, N // 2),
        )

        self.h_s = nn.Sequential(
            nn.Linear(N // 2, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N * 3 // 2, 2 * N),
        )

        self.entropy_bottleneck = EntropyBottleneck(N // 2)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_scales = nn.Parameter(
            20 * torch.ones((n_levels, N), requires_grad=True), requires_grad=True
        )
        self.h_scales = nn.Parameter(
            20 * torch.ones((n_levels, N // 2), requires_grad=True), requires_grad=True
        )

    def scale_latents(self, y: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            g_scale = (torch.abs(self.g_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.g_scales[int(level) + 1]) ** interp
            )

        else:
            g_scale = torch.abs(self.g_scales[int(level)])

        return y * g_scale.unsqueeze(0)

    def descale_latents(self, y_hat: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            g_scale = (torch.abs(self.g_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.g_scales[int(level) + 1]) ** interp
            )
        else:
            g_scale = torch.abs(self.g_scales[int(level)])

        return y_hat / g_scale.unsqueeze(0)

    def scale_hyper_latents(self, z: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            h_scale = (torch.abs(self.h_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.h_scales[int(level) + 1]) ** interp
            )

        else:
            h_scale = torch.abs(self.h_scales[int(level)])

        return z * h_scale.unsqueeze(0)

    def descale_hyper_latents(self, z_hat: torch.Tensor, level: int):

        if level != int(level):
            interp = level - int(level)
            h_scale = (torch.abs(self.h_scales[int(level)]) ** (1 - interp)) * (
                torch.abs(self.h_scales[int(level) + 1]) ** interp
            )
        else:
            h_scale = torch.abs(self.h_scales[int(level)])

        return z_hat / h_scale.unsqueeze(0)

    def forward(
        self, x_prev: torch.Tensor, x_cur: torch.Tensor, level: int, training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """CodecNet forward pass.

        Args:
            x_prev (torch.Tensor): Gaussians [N, C] from previous level, multiplied by \alpha.
            x_cur (torch.Tensor): Gaussians [N, C] from current level, multiplied by \alpha.

        Returns:
            y: Compressed and corrective Gaussians [N, C].
        """
        x_cat = torch.cat((x_prev, x_cur), dim=1)

        y = self.g_a(x_cat)
        y_scaled = self.scale_latents(y, level)

        z = self.h_a(y_scaled)
        z_scaled = self.scale_hyper_latents(z, level)
        z_permute = torch.permute(z_scaled, (1, 0)).unsqueeze(0)  # [1, C, N]

        z_hat, z_likelihoods = self.entropy_bottleneck(z_permute, training=training)

        z_hat_permute = z_hat.squeeze(0).permute(1, 0)  # [N, C]
        z_descaled = self.descale_hyper_latents(z_hat_permute, level)

        entropy_params = self.h_s(z_descaled)
        scales_hat, means_hat = entropy_params.chunk(2, 1)

        scales_hat_permute = torch.permute(scales_hat, (1, 0)).unsqueeze(0)
        means_hat_permute = torch.permute(means_hat, (1, 0)).unsqueeze(0)

        y_permute = y_scaled.permute(1, 0).unsqueeze(0)  # [1, C, N]

        compressed_y, y_likelihoods = self.gaussian_conditional(
            y_permute, scales_hat_permute, means=means_hat_permute, training=training
        )
        compressed_y_permute = compressed_y.squeeze(0).permute(1, 0)  # [N, C]
        y_descaled = self.descale_latents(compressed_y_permute, level)

        output = self.g_s(y_descaled)

        mask, warping_params = output.chunk(2, 1)

        return (torch.sigmoid(mask), warping_params), {"y": y_likelihoods, "z": z_likelihoods}


class ConditionalGSCodec(CompressionModel):
    def __init__(self, dataset: Namespace, M: int, N: int | None = None, n_levels: int = 1):
        """Conditional Gaussian Splat Codec.
        Performs both intra-coding and inter-coding of the attributes of the splats.

        Inspired from:
        https://github.com/Orange-OpenSource/AIVC/tree/3ce96717ea749d70b308a4ec360f1bb943405c14

        Args:
            M (int): Bottleneck dimensionality.
            N (int, optional): Not used for this compressor. Defaults to None. Still exists
                to keep the same API with other compressors.
        """
        super(ConditionalGSCodec, self).__init__()

        self.n_levels = n_levels

        self.codec_net = CodecNet(dataset, M, N, n_levels)
        self.msg_net = MSGNet(dataset, M, N, n_levels)

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

    @classmethod
    def calculate_entropy(self, likelihoods: Dict[str, Dict]) -> torch.Tensor:
        codec_likelihood = sum(
            (torch.log(lhoods).sum() / (-math.log(2))) for lhoods in likelihoods["codec"].values()
        )
        msg_likelihood = sum(
            (torch.log(lhoods).sum() / (-math.log(2))) for lhoods in likelihoods["msg"].values()
        )
        return codec_likelihood, msg_likelihood

    def get_cut_attributes(
        self,
        gaussians: GaussianModel,
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

        num_cubes = 2 ** chosen_depth
        node_assignments = build_octree(gaussians.get_xyz, num_cubes)

        mu = gaussians.get_xyz.contiguous()
        scaling = gaussians.get_scaling.contiguous()
        rotation = gaussians.get_rotation.contiguous()
        sigma = gaussians.get_covariance().contiguous()
        opacity = gaussians.get_opacity.contiguous()
        sh = gaussians.get_features.contiguous()

        weights = calculate_weights(sigma, opacity).contiguous()

        # # Call function to perform aggregation
        # node_xyzs, node_sigma, node_opacity, node_shs = AggregationFunction.apply(
        #     weights,
        #     mu,
        #     sigma,
        #     opacity,
        #     sh.view(sh.shape[0], -1),
        #     node_assignments.type(torch.int32),
        # )

        # Number of groups
        num_nodes = torch.max(node_assignments).item() + 1

        # Aggregate weights
        node_weights = torch.zeros(num_nodes, 1).cuda()
        node_weights.scatter_add_(0, node_assignments.unsqueeze(1), weights)
        
        # Aggregate means
        _gaussian_means = weights * mu
        node_xyzs = torch.zeros(num_nodes, _gaussian_means.shape[1]).cuda()
        node_xyzs = node_xyzs.scatter_add(
            0, 
            node_assignments.unsqueeze(1).expand(-1, _gaussian_means.shape[1]), 
            _gaussian_means
        ) / node_weights

        # Aggregate scaling
        _gaussian_scaling = weights * scaling
        node_scaling = torch.zeros(num_nodes, _gaussian_scaling.shape[1]).cuda()
        node_scaling = node_scaling.scatter_add(
            0,
            node_assignments.unsqueeze(1).expand(-1, _gaussian_scaling.shape[1]),
            _gaussian_scaling
        ) / node_weights

        # Aggregate rotation
        _gaussian_rotation = weights * rotation
        node_rotation = torch.zeros(num_nodes, _gaussian_rotation.shape[1]).cuda()
        node_rotation = node_rotation.scatter_add(
            0,
            node_assignments.unsqueeze(1).expand(-1, _gaussian_rotation.shape[1]),
            _gaussian_rotation
        ) / node_weights

        # Aggregate sigma
        _gaussian_sigma = weights * sigma
        node_sigma = torch.zeros(num_nodes, _gaussian_sigma.shape[1]).cuda()

        diff_mu = mu - node_xyzs[node_assignments]
        diff_mu_outer = torch.einsum("ij,ik->ijk", diff_mu, diff_mu)
        diff_mu_vector = pack_full_to_lower_triangular(diff_mu_outer)
        _gaussian_sigma = _gaussian_sigma + weights * diff_mu_vector

        node_sigma = torch.zeros(num_nodes, _gaussian_sigma.shape[1]).cuda()
        node_sigma = node_sigma.scatter_add(
            0,
            node_assignments.unsqueeze(1).expand(-1, _gaussian_sigma.shape[1]),
            _gaussian_sigma
        ) / node_weights

        # Aggregate opacity
        _gaussian_opacity = weights * opacity
        node_opacity = torch.zeros(num_nodes, 1).cuda()
        node_opacity = node_opacity.scatter_add(
            0,
            node_assignments.unsqueeze(1).expand(-1, _gaussian_opacity.shape[1]),
            _gaussian_opacity
        ) / node_weights

        # Aggregate SH
        _gaussian_shs = weights * sh.view(sh.shape[0], -1)
        node_shs = torch.zeros(num_nodes, _gaussian_shs.shape[1]).cuda()
        node_shs = node_shs.scatter_add(
            0,
            node_assignments.unsqueeze(1).expand(-1, _gaussian_shs.shape[1]),
            _gaussian_shs
        ) / node_weights

        return (
            node_xyzs,
            node_scaling,
            node_rotation,
            node_sigma,
            node_shs.view(node_shs.shape[0], -1, 3),
            node_opacity,
            node_assignments.type(torch.int32)
        )

    def apply_warping(self, x_prev: torch.Tensor, warping_params: torch.Tensor) -> torch.Tensor:
        """Apply warping to the previous Gaussians.

        Args:
            x_prev (torch.Tensor): Gaussians [N, C] from previous level, multiplied by \alpha.
            warping_params (torch.Tensor): Warping parameters [N, C].

        Returns:
            x_pred: Predicted Gaussians [N, C].
        """
        warped_output = torch.zeros_like(x_prev)

        additive_position = warping_params[:, :3]
        scale, quaternion = warping_params[:, 3:6], warping_params[:, 6:10]
        additive_opacity = warping_params[:, 10:11]
        additive_sh = warping_params[:, 11:]

        # warped_scale = torch.log((1.0e-6 + torch.sigmoid(scale)) * torch.exp(x_prev[:, 3:6]))
        # normalized_quaternion = F.normalize(quaternion)
        # warped_quaternion = (
        #     normalized_quaternion
        #     * F.normalize(x_prev[:, 6:10])
        #     * conjugate_quaternion(normalized_quaternion)
        # )
        # warped_output[:, 3:6] = warped_scale
        # warped_output[:, 6:10] = warped_quaternion
        
        warped_output[:, 3:6] = torch.abs(scale + x_prev[:, 3:6])
        warped_output[:, 6:10] = torch.nn.functional.normalize(
            quaternion + x_prev[:, 6:10])
        
        # print(additive_position.min(), additive_position.max())
        # print(warped_scale.min(), warped_scale.max())
        # print(warped_quaternion.min(), warped_quaternion.max())
        # print(additive_opacity.min(), additive_opacity.max())
        # print(additive_sh.min(), additive_sh.max())
        # breakpoint()

        warped_output[:, :3] = x_prev[:, :3] + additive_position
       
        warped_output[:, 10:11] = x_prev[:, 10:11] + additive_opacity
        warped_output[:, 11:] = x_prev[:, 11:] + additive_sh

        return warped_output

    def forward(
        self,
        x_prev: torch.Tensor,
        x_cur: torch.Tensor,
        level: int,
        intra: bool = False,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """ConditionalGSCodec forward pass.

        Args:
            x_prev (torch.Tensor): Gaussians [N, C] from previous level, multiplied by \alpha.
            x_cur (torch.Tensor): Gaussians [N, C] from current level, multiplied by \alpha.

        Returns:
            y: Compressed and corrective Gaussians [N, C].
        """
        if not intra:
            (mask, warping_params), msg_likelihoods = self.msg_net(x_prev, x_cur, level, training)
            x_pred = self.apply_warping(x_prev, warping_params)
            x_complement, codec_likelihoods = self.codec_net(
                mask * x_pred, mask * x_cur, level, training
            )
            x_final = (1 - mask) * x_pred + mask * x_complement
        else:
            msg_likelihoods = {"y": torch.ones(1).cuda(), "z": torch.ones(1).cuda()}
            mask = torch.ones_like(x_prev)
            x_complement, codec_likelihoods = self.codec_net(mask * x_prev, mask * x_cur, level)
            x_final = x_complement

        return x_final, {"codec": codec_likelihoods, "msg": msg_likelihoods}

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
