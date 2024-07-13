from typing import Tuple

# import octree_final as hierarchy_generation
import build_octree as hierarchy_generation
import torch

from models.splatting.mcmc_model import GaussianModel
from scene.cameras import Camera


def choose_min_max_depth(points3D) -> Tuple[int, int]:
    """Choose the minimum and maximum depth for the octree based on the 3D points.
    The minimum depth is calculated based on the maximum bounding box size of the 3D points.

    Args:
        points3D (torch.Tensor): The 3D points.

    Returns:
        Tuple[int, int]: The minimum and maximum depth of the octree.
    """
    aabb_min = torch.min(points3D, dim=0)[0]
    aabb_max = torch.max(points3D, dim=0)[0]

    box_d = aabb_max - aabb_min
    max_depth = int(torch.ceil(torch.log2(torch.max(box_d) / 0.01)).item())
    min_depth = max_depth - 4
    # min_depth = int(torch.ceil(torch.log2(torch.max(box_d) / 0.5)).item())
    # max_depth = min_depth + 5
    return min_depth, max_depth


def build_octree(point3D: torch.Tensor, max_depth: int):
    """Builds an octree from the given 3D points and return the node assignment of each 3D point
    to one of the intermediate nodes

    Args:
        point3D (torch.Tensor): The 3D points to build the octree from
        max_depth (int, optional): The maximum depth of the octree. Defaults to 20.

    Returns:
        torch.Tensor: The node assignment of each 3D point to one of the intermediate nodes
            with shape (N, 1)
    """
    with torch.no_grad():
        aabb_min = torch.min(point3D, dim=0)[0]
        aabb_max = torch.max(point3D, dim=0)[0]

        box_d = aabb_max - aabb_min
        box_min = (aabb_min - 0.01 * box_d).contiguous()
        box_max = (aabb_max + 0.01 * box_d).contiguous()

        point3D = point3D.contiguous()

        octree = hierarchy_generation.Octree(max_depth)
        new_indices = octree.get_point_node_assignment(point3D, box_min, box_max)

        # cube_size = torch.max(box_max - box_min) / num_cubes

        # # Compute the grid indices for each point
        # indices = ((point3D - box_min) / cube_size).floor().long()

        # # Convert 3D indices to a unique group ID
        # # Assuming the grid extends only within the AABB:
        # grid_dim = ((box_max - box_min) / cube_size).ceil().long()
        # group_ids = indices[:, 0] + indices[:, 1] * grid_dim[0] + indices[:, 2] * grid_dim[0] * grid_dim[1]

        # _, new_indices = torch.unique(group_ids, sorted=True, return_inverse=True)
    return new_indices


# def select_tree_cut(
#     built_octree: hierarchy_generation.Octree,
#     camera: Camera,
#     granularity_threshold: float = 2.0,
#     chosen_depth: int = -1
# ):
#     with torch.no_grad():
#         node_indices = built_octree.select_tree_cut(
#             camera.full_proj_transform.contiguous(),
#             camera.world_view_transform.contiguous(),
#             int(camera.image_height),
#             int(camera.image_width),
#             granularity_threshold,
#             chosen_depth
#         )

#     return node_indices

# def calculate_weights(scales, opacity) -> torch.Tensor:
#     """Calculates the aggregation weights of Gaussians. This calculation is based on the
#     3-sigma surface area of Gaussians. The formula is as follows:
#     $$4*\pi*((a^p*b^p + a^p*c^p + b^p*c^p)/3)^{1/p}$$

#     Args:
#         scales (torch.Tensor): The scales of the Gaussians.
#         opacity (torch.Tensor): The opacity of the Gaussians.

#     Returns:
#         torch.Tensor: The weights of the Gaussians.
#     """
#     radii = 3 * scales
#     p = 1.6075
#     a, b, c = radii.chunk(3, dim=1)
#     # 4 * torch.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3)**(1/p)
#     weights = opacity * ((a**p * b**p + a**p * c**p + b**p * c**p) ** (1 / p))
#     return weights

EPSILON = 1e-8


def determinant3x3(covariances: torch.Tensor) -> torch.Tensor:
    """Calculates the determinant of a batch of 3x3 matrices represented by the upper
    triangular part

    Args:
        covariances (torch.Tensor): The upper triangular part of 3x3 matrices (batch_size, 6)

    Returns:
        torch.Tensor: The determinant of the 3x3 matrices
    """
    # Assuming covariances is a tensor of shape (batch_size, 6) representing the upper triangular part of 3x3 matrices
    a = covariances[:, 0]
    b = covariances[:, 1]
    c = covariances[:, 2]
    d = covariances[:, 1]  # d is the same as b
    e = covariances[:, 3]
    f = covariances[:, 4]
    g = covariances[:, 2]  # g is the same as c
    h = covariances[:, 4]  # h is the same as f
    i = covariances[:, 5]

    # Calculate the determinant of the 3x3 matrices for the batch
    determinants = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return determinants


def calculate_weights(covariance: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    """Calculates the weights of the Gaussians based on the 3-sigma surface area of the Gaussians."""

    # Calculate the determinant of the covariance matrix
    determinant = determinant3x3(covariance)
    determinant = torch.abs(determinant) + EPSILON

    # Calculate the square root of the determinant
    characteristic_length = torch.pow(determinant, 1 / 3)

    # Use a proportional factor to approximate the surface area
    # The proportional factor here is arbitrary, chosen to match typical scaling
    # proportional_factor = 4.836  # This factor is to adjust the scale, it's not exact
    # surface_area_approx = proportional_factor * characteristic_length
    surface_area_approx = characteristic_length

    return opacity * surface_area_approx[:, None]


def assign_unique_values(data):
    # data should be a CUDA tensor of shape (N, 10)
    new_data = data.clone()
    N, C = data.shape
    for i in range(C):
        unique_values, new_indices = torch.unique(data[:, i], sorted=True, return_inverse=True)
        new_data[:, i] = new_indices  # Replace column with new indices

    return new_data


class AggregationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, position, sigma, opacity, shs, node_ids):
        """Forward pass of the aggregation function. This function aggregates the Gaussians
        based on the node assignments. The aggregation is done on low-level CUDA code for
        efficiency and parallelization.
        """
        num_gaussians = position.shape[0]
        num_nodes_at_level = (node_ids.max() + 1).item()

        new_position, new_sigma, new_opacity, new_shs, new_node_total_weights = (
            hierarchy_generation.aggregate_gaussians_fwd(
                weights,
                position,
                sigma,
                opacity,
                shs,
                node_ids,
                num_nodes_at_level,
            )
        )

        # Store for backward
        ctx.save_for_backward(
            weights,
            position,
            sigma,
            opacity,
            shs,
            new_position,
            new_sigma,
            node_ids,
            new_node_total_weights,
        )
        ctx.num_gaussians = num_gaussians
        ctx.num_nodes_at_level = num_nodes_at_level

        return new_position, new_sigma, new_opacity, new_shs

    @staticmethod
    def backward(ctx, grad_new_position, grad_new_sigma, grad_new_opacity, grad_new_shs):
        """NOTE: Backward is currently not utilized and requires additional attention for
        implementation.
        """
        # Retrieve saved tensors
        (
            weights,
            position,
            sigma,
            opacity,
            sh,
            new_position,
            new_sigma,
            node_ids,
            node_total_weights,
        ) = ctx.saved_tensors

        num_gaussians = ctx.num_gaussians
        num_nodes_at_level = ctx.num_nodes_at_level

        # Calculate gradients for inputs based on grad_outputs
        grad_weights, grad_position, grad_sigma, grad_opacity, grad_sh = (
            hierarchy_generation.aggregate_gaussians_bwd(
                grad_new_position,
                grad_new_sigma,
                grad_new_opacity,
                grad_new_shs,
                node_total_weights,
                weights,
                position,
                sigma,
                opacity,
                sh,
                new_position,
                new_sigma,
                node_ids,
                num_gaussians,
                num_nodes_at_level,
            )
        )

        return None, grad_position, grad_sigma, grad_opacity, grad_sh, None


def aggregate_gaussians_recursively(
    weights: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    opacity: torch.Tensor,
    sh: torch.Tensor,
    node_ids: torch.Tensor,
    min_level: int,
    max_level: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregates the Gaussians in a hierarchical manner. The aggregation is done from the
    lowest level to the highest level in a recursive manner.

    Args:
        weights (torch.Tensor): The weights of the Gaussians
        mu (torch.Tensor): The mean of the Gaussians
        sigma (torch.Tensor): The covariance of the Gaussians
        opacity (torch.Tensor): The opacity of the Gaussians
        sh (torch.Tensor): The spherical harmonics of the Gaussians
        node_ids (torch.Tensor): The node assignment of each Gaussian to one of the intermediate nodes
        min_level (int): The minimum level to aggregate from
        max_level (int): The maximum level to aggregate to

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The aggregated mean,
            covariance, opacity, and spherical harmonics of the Gaussians
    """

    assert min_level <= max_level, "Minimum level should be less than or equal to maximum level"

    current_mu = mu
    current_opacity = opacity
    current_sh = sh
    current_weights = weights
    current_sigma = sigma

    # Current mapping between Gaussians and lowest level nodes
    sorted_node_ids = node_ids

    for level_idx, level in enumerate(range(max_level - 1, min_level - 1, -1)):
        # Map node ids to dense structure instead of sparse one
        new_node_ids_assigned = sorted_node_ids[:, level].type(torch.int32)

        # Call function to perform aggregation
        new_mu, new_sigma, new_opacity, new_sh = AggregationFunction.apply(
            current_weights,
            current_mu,
            current_sigma,
            current_opacity,
            current_sh,
            new_node_ids_assigned,
        )

        current_mu = new_mu
        current_sigma = new_sigma
        current_opacity = new_opacity[:, None]
        current_sh = new_sh

        if level > min_level:
            current_weights = calculate_weights(current_sigma, current_opacity)

            # Sort the node id assignment matrix according to the order of new node ids
            sorted_node_ids = sorted_node_ids[new_node_ids_assigned.sort()[1]]

            # Find where the value changes in the specified column after sorting
            unique_indices = torch.cat(
                (torch.tensor([True]).cuda(), sorted_node_ids[1:, -1] != sorted_node_ids[:-1, -1])
            )
            # Discard the last column as it is not needed
            sorted_node_ids = sorted_node_ids[unique_indices, :-1]

    return (current_mu, current_sigma, current_opacity, current_sh)
