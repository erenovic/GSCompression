#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""THIS CODE IS NOT UTILIZED IN THE CURRENT REPOSITORY AND KEPT FOR FUTURE REFERENCE"""

import logging
import os
from argparse import Namespace

import numpy as np
import tinycudann as tcnn
import torch
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from tqdm import tqdm

from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import l2_loss
from utils.reloc_utils import compute_relocation_cuda
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, dataset):
        self.active_sh_degree = 0
        self.max_sh_degree = dataset.sh_degree

        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._encoding_features = torch.empty(0)
        self.scene_extent = torch.empty(0)

        self.optimizer = None
        self.setup_functions()

        encoding_num_levels = dataset.position_enc["num_levels"]
        encoding_features_per_level = dataset.position_enc["features_per_level"]
        self.position_enc_feature_dim = encoding_num_levels * encoding_features_per_level
        self.feature_dim = dataset.feature_dim
        self.hidden_dim = dataset.hidden_dim
        self.position_enc_feature_dim = encoding_num_levels * encoding_features_per_level
        self._position_encoding = HashEncoding(
            num_levels=encoding_num_levels,
            min_res=dataset.position_enc["min_resolution"],
            max_res=dataset.position_enc["max_resolution"],
            log2_hashmap_size=dataset.position_enc["log2_hashmap_size"],
            features_per_level=encoding_features_per_level,
            implementation="tcnn",
        ).cuda()

        # Define the encoding configuration
        encoding_config = {
            "otype": "Frequency",  # Type of encoding, e.g., Fourier or SphericalHarmonics
            "n_frequencies": 4,  # Number of frequency bands
        }
        n_input_dims = 3
        self._direction_encoding = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config=encoding_config,
        )
        self.direction_enc_feature_dim = n_input_dims * encoding_config["n_frequencies"] * 2
        # self.embedding_dim = self.feature_dim + position_enc_feature_dim + 3 + 1
        self.init_neural_networks()

    def init_neural_networks(self):
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feature_dim + self.position_enc_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).cuda()
        self.mlp_covariance = nn.Sequential(
            nn.Linear(self.feature_dim + self.position_enc_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 7),
        ).cuda()
        self.mlp_color = nn.Sequential(
            nn.Linear(
                self.feature_dim
                + self.position_enc_feature_dim
                + self.direction_enc_feature_dim
                + 1,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        ).cuda()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._encoding_features,
            self._position_encoding.state_dict(),
            self.mlp_opacity.state_dict(),
            self.mlp_covariance.state_dict(),
            self.mlp_color.state_dict(),
            self.scene_extent,
            self.max_radii2D,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._encoding_features,
            position_enc_dict,
            mlp_opacity_dict,
            mlp_covariance_dict,
            mlp_color_dict,
            self.scene_extent,
            self.max_radii2D,
            opt_dict,
            spatial_lr_scale,
        ) = model_args

        if training_args:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
        else:
            print("No training setup provided")

        self._position_encoding.load_state_dict(position_enc_dict)
        self.mlp_opacity.load_state_dict(mlp_opacity_dict)
        self.mlp_covariance.load_state_dict(mlp_covariance_dict)
        self.mlp_color.load_state_dict(mlp_color_dict)

    def set_initial_neural_network_parameters(self, scales, rots, opacities, colors):
        N_ITER = 200
        progress_bar = tqdm(range(N_ITER), desc="Warming-up networks...")
        ema_loss_for_log = 0.0
        ema_extent_for_log = 0.0

        self.train()

        l = [
            {
                "params": [self._encoding_features],
                "lr": 0.0075,
                "name": "encoding_features",
            },
            {
                "params": self.mlp_opacity.parameters(),
                "lr": 0.002,
                "name": "mlp_opacity",
            },
            {
                "params": self.mlp_covariance.parameters(),
                "lr": 0.004,
                "name": "mlp_cov",
            },
            {
                "params": self.mlp_color.parameters(),
                "lr": 0.008,
                "name": "mlp_color",
            },
            {
                "params": self._position_encoding.parameters(),
                "lr": 0.001,
                "name": "position_encoding",
            },
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        for iteration in range(N_ITER):
            scaling, rotation, opacity, extent, features = self.get_shape_parameters()
            color = self.get_color_parameters(self.get_xyz, None, features)

            extent_loss: torch.Tensor = l2_loss(extent[:, 0], scales.max(dim=1)[0])
            rot_loss: torch.Tensor = l2_loss(rotation, rots)
            scale_loss: torch.Tensor = l2_loss(scaling, scales)
            color_loss: torch.Tensor = l2_loss(color, colors)
            opacity_loss: torch.Tensor = l2_loss(opacity, opacities)

            loss = rot_loss + scale_loss + color_loss + opacity_loss + extent_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_extent_for_log = 0.4 * extent.mean().item() + 0.6 * ema_extent_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.5f}", "Extent": f"{ema_extent_for_log:.2f}"}
                )
                progress_bar.update(10)

        self.init_opacity_parameters = self.mlp_opacity.state_dict()

    @property
    def get_xyz(self):
        return self._xyz

    def get_opacity_and_scaling(self, idxs):
        if idxs is None:
            idxs = torch.arange(self.get_xyz.shape[0], device="cuda")

        encoded_position = self._position_encoding(self.get_xyz[idxs] / self.scene_extent).type(
            torch.float32
        )

        features = torch.cat(
            (self._encoding_features[idxs], encoded_position),
            dim=1,
        )
        opacity = torch.sigmoid(self.mlp_opacity(features))
        covariance = self.mlp_covariance(features)
        scaling = self.scaling_activation(self._scaling[idxs]) * torch.sigmoid(covariance[:, :3])
        return opacity, scaling

    def get_shape_parameters_from_features(self, input_features, extent):
        covariance = self.mlp_covariance(input_features)
        opacity = torch.sigmoid(self.mlp_opacity(input_features))
        scaling = extent * torch.sigmoid(covariance[:, :3])
        rotation = self.rotation_activation(covariance[:, 3:])

        return scaling, rotation, opacity, extent

    def get_shape_parameters(self):
        encoded_position = self._position_encoding(self.get_xyz / self.scene_extent).type(
            torch.float32
        )
        # encoding_features = torch.zeros((self.get_xyz.shape[0], self.feature_dim), device="cuda")
        features = torch.cat(
            (self._encoding_features, encoded_position),
            dim=1,
        )
        return (
            *self.get_shape_parameters_from_features(
                features, self.scaling_activation(self._scaling)
            ),
            features,
        )

    def get_color_parameters(self, position, camera_center, input_features):
        if camera_center is None:
            camera_dir = torch.zeros_like(position)
            camera_dist = torch.zeros((position.shape[0], 1), device="cuda")
        else:
            camera_dir = position - camera_center
            camera_dist = torch.norm(camera_dir, dim=1, keepdim=True)
            camera_dir = camera_dir / camera_dist

        camera_dir_features = self._direction_encoding(camera_dir)
        color_features = torch.cat((input_features, camera_dir_features, camera_dist), dim=1)
        color = torch.sigmoid(self.mlp_color(color_features))
        return color

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        tensor_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        fused_color = RGB2SH(tensor_color)
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        logging.info("Number of points at initialisation: " + str(fused_point_cloud.shape[0]))

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = 0.1 * torch.ones(
            (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._encoding_features = nn.Parameter(
            torch.zeros((fused_point_cloud.shape[0], self.feature_dim), device="cuda")
        )
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scales[:, :1]).requires_grad_(True)
        )

        # TODO: Now I am fixing scene extent to 1.5 times the max extent of the point cloud
        # This might not be correct based on SfM accuracy!
        with torch.no_grad():
            min_bbox = torch.min(fused_point_cloud, dim=0).values
            max_bbox = torch.max(fused_point_cloud, dim=0).values
            self.scene_extent = torch.max(max_bbox - min_bbox) * 2
            self.scene_extent.requires_grad_(False)

        self.set_initial_neural_network_parameters(scales, rots, opacities, tensor_color)

    def training_setup(self, optimization_args: Namespace):
        self.percent_dense = optimization_args.percent_dense

        l = [
            {
                "params": [self._xyz],
                "lr": optimization_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._scaling],
                "lr": optimization_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._encoding_features],
                "lr": optimization_args.feature_lr,
                "name": "encoding_features",
            },
            {
                "params": self.mlp_opacity.parameters(),
                "lr": optimization_args.mlp_opacity_lr_init,
                "name": "mlp_opacity",
            },
            {
                "params": self.mlp_covariance.parameters(),
                "lr": optimization_args.mlp_cov_lr_init,
                "name": "mlp_cov",
            },
            {
                "params": self.mlp_color.parameters(),
                "lr": optimization_args.mlp_color_lr_init,
                "name": "mlp_color",
            },
            {
                "params": self._position_encoding.parameters(),
                "lr": optimization_args.position_encoding_lr,
                "name": "position_encoding",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=optimization_args.position_lr_init * self.spatial_lr_scale,
            lr_final=optimization_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=optimization_args.position_lr_delay_mult,
            max_steps=optimization_args.position_lr_max_steps,
        )
        self.mlp_opacity_scheduler_args = get_expon_lr_func(
            lr_init=optimization_args.mlp_opacity_lr_init,
            lr_final=optimization_args.mlp_opacity_lr_final,
            lr_delay_mult=optimization_args.mlp_opacity_lr_delay_mult,
            max_steps=optimization_args.mlp_opacity_lr_max_steps,
        )
        self.mlp_cov_scheduler_args = get_expon_lr_func(
            lr_init=optimization_args.mlp_cov_lr_init,
            lr_final=optimization_args.mlp_cov_lr_final,
            lr_delay_mult=optimization_args.mlp_cov_lr_delay_mult,
            max_steps=optimization_args.mlp_cov_lr_max_steps,
        )
        self.mlp_color_scheduler_args = get_expon_lr_func(
            lr_init=optimization_args.mlp_color_lr_init,
            lr_final=optimization_args.mlp_color_lr_final,
            lr_delay_mult=optimization_args.mlp_color_lr_delay_mult,
            max_steps=optimization_args.mlp_color_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name: str):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            if ("mlp" in group["name"]) or ("position_encoding" in group["name"]):
                continue

            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            if ("mlp" in group["name"]) or ("position_encoding" in group["name"]):
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, reset_params=True):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._encoding_features = optimizable_tensors["encoding_features"]
        self._scaling = optimizable_tensors["scaling"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            if ("mlp" in group["name"]) or ("position_encoding" in group["name"]):
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        old_opacity, old_scaling = self.get_opacity_and_scaling(idxs)
        new_opacity, new_scaling = compute_relocation_cuda(
            # opacity_old=self.get_opacity[idxs, 0],
            opacity_old=old_opacity,
            scale_old=old_scaling,
            N=ratio[idxs, 0] + 1,
        )
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005
        )
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return (
            self._xyz[idxs],
            self._features_dc[idxs],
            self._features_rest[idxs],
            new_opacity,
            new_scaling,
            self._rotation[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        (
            self._xyz[dead_indices],
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def radsplat_prune(self, max_alphas, radsplat_prune_threshold: float):
        prune_mask = (max_alphas < radsplat_prune_threshold).squeeze()
        self.prune_points(prune_mask, reset_params=True)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation) = (
            self._update_params(add_idx, ratio=ratio)
        )

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            reset_params=False,
        )
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs
