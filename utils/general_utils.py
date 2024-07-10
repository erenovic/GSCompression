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

import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def unpack_lower_triangular_to_full(mat: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
    """Unpacks the lower triangular part of matrices into full symmetric matrices.

    Args:
        mat (torch.Tensor): Tensor of shape (N, 6) representing the packed lower triangular part of 3x3 matrices.

    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) of full matrices.
    """
    N = mat.shape[0]
    full_matrices = torch.zeros((N, 3, 3), device=mat.device, dtype=mat.dtype)

    # Fill the diagonal
    full_matrices[:, 0, 0] = mat[:, 0]
    full_matrices[:, 1, 1] = mat[:, 3]
    full_matrices[:, 2, 2] = mat[:, 5]

    # Fill the lower triangular part
    full_matrices[:, 1, 0] = mat[:, 1]
    full_matrices[:, 2, 0] = mat[:, 2]
    full_matrices[:, 2, 1] = mat[:, 4]

    if symmetric:
        # Reflect the lower triangular to the upper triangular part
        full_matrices[:, 0, 1] = mat[:, 1]
        full_matrices[:, 0, 2] = mat[:, 2]
        full_matrices[:, 1, 2] = mat[:, 4]
    return full_matrices


def pack_full_to_lower_triangular(mat: torch.Tensor) -> torch.Tensor:
    """Packs the lower triangular part of matrices into full symmetric matrices.

    Args:
        mat (torch.Tensor): Tensor of shape (N, 3, 3) representing the full symmetric matrices.

    Returns:
        torch.Tensor: Tensor of shape (N, 6) of packed lower triangular part of 3x3 matrices.
    """
    N = mat.shape[0]
    lower_triangular = torch.zeros((N, 6), device=mat.device, dtype=mat.dtype)

    # Pack the diagonal
    lower_triangular[:, 0] = mat[:, 0, 0]
    lower_triangular[:, 3] = mat[:, 1, 1]
    lower_triangular[:, 5] = mat[:, 2, 2]

    # Pack the lower triangular part
    lower_triangular[:, 1] = mat[:, 1, 0]
    lower_triangular[:, 2] = mat[:, 2, 0]
    lower_triangular[:, 4] = mat[:, 2, 1]
    return lower_triangular


def rotation_matrix_to_quaternion(rot_matrices):
    # Preallocate quaternion tensor
    B = rot_matrices.shape[0]
    quaternions = torch.empty((B, 4), dtype=torch.float32, device=rot_matrices.device)

    # Components from the rotation matrix diagonal
    m00 = rot_matrices[:, 0, 0]
    m11 = rot_matrices[:, 1, 1]
    m22 = rot_matrices[:, 2, 2]

    # Calculate each component of the quaternion
    quaternions[:, 0] = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0))  # a
    quaternions[:, 1] = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)),
        rot_matrices[:, 2, 1] - rot_matrices[:, 1, 2],
    )  # b
    quaternions[:, 2] = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)),
        rot_matrices[:, 0, 2] - rot_matrices[:, 2, 0],
    )  # c
    quaternions[:, 3] = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)),
        rot_matrices[:, 1, 0] - rot_matrices[:, 0, 1],
    )  # d

    # Normalize the quaternion to ensure it is a unit quaternion
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)

    return quaternions


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def conjugate_quaternion(quaternions):
    """
    Compute the conjugate of a batch of quaternions.

    Args:
    quaternions: torch.Tensor of shape (B, 4) where each quaternion is represented as (w, x, y, z)

    Returns:
    conjugated_quaternions: torch.Tensor of shape (B, 4) representing the conjugated quaternions
    """
    # Negate the vector part (last three components) and keep the scalar part (first component) the same
    conjugated_quaternions = quaternions.clone()
    conjugated_quaternions[:, 1:] = -conjugated_quaternions[:, 1:]

    return conjugated_quaternions


def nearest_psd(A, tol=1e-8):
    """
    Find the nearest positive semi-definite matrix using the Higham algorithm.

    Args:
        A (torch.Tensor): Tensor of shape (N, 3, 3) representing N covariance matrices.
        tol (float): Tolerance for the stopping criterion.

    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) with positive semi-definite covariance matrices.
    """
    N, m, n = A.shape
    assert m == n, "Each matrix must be square."

    def proj_s(A):
        """
        Projection onto the space of symmetric matrices.
        """
        return (A + A.transpose(-2, -1)) / 2

    def proj_psd(A):
        """
        Projection onto the space of positive semi-definite matrices.
        """
        eigval, eigvec = torch.linalg.eigh(A)
        eigval = torch.clamp(eigval, min=0)
        return eigvec @ torch.diag_embed(eigval) @ eigvec.transpose(-2, -1)

    Y = A.clone()
    for i in range(100):  # Maximum number of iterations
        R = Y - proj_psd(Y - A)
        Y = proj_s(R)
        if torch.norm(A - Y, p='fro') / torch.norm(A, p='fro') < tol:
            break

    return Y


def make_psd(cov_matrices: torch.Tensor) -> torch.Tensor:
    """
    Ensure the input covariance matrices are positive semi-definite.

    Args:
        cov_matrices (torch.Tensor): Tensor of shape (N, 6) representing N covariance matrices.

    Returns:
        torch.Tensor: Tensor of shape (N, 6) with positive semi-definite covariance matrices.
    """

    # Unpack the lower triangular part of the covariance matrices
    full_cov = unpack_lower_triangular_to_full(cov_matrices)

    # Symmetric/hermitian decomposition
    eig_vals, eig_vecs = torch.linalg.eigh(full_cov)

    # Clip negative eigenvalues to zero
    eig_vals_clipped = torch.clamp(eig_vals, min=0.0)

    # Reconstruct the covariance matrices
    reconstructed_matrices = torch.matmul(
        eig_vecs, torch.matmul(torch.diag_embed(eig_vals_clipped), eig_vecs.transpose(-2, -1))
    )
    # reconstructed_matrices = nearest_psd(full_cov)

    # Pack the lower triangular part of the covariance matrices
    return strip_lowerdiag(reconstructed_matrices)


def decompose_covariance_matrix(covariances):
    # # Perform SVD on the covariance matrices
    # U, S, Vt = torch.svd(unpack_lower_triangular_to_full(covariances))

    # # The rotation matrix R is obtained by multiplying U and Vt
    # R = torch.matmul(U, Vt)

    # Q = rotation_matrix_to_quaternion(R)

    full_covs = unpack_lower_triangular_to_full(covariances)

    L, Q = torch.linalg.eigh(full_covs)

    L_clamped = torch.sqrt(torch.clamp(L, min=0))
    L_final = torch.clamp(L_clamped, min=1e-6)

    Q_quaternion = rotation_matrix_to_quaternion(Q)

    return L_final, Q_quaternion


def compute_required_rotation_and_scaling(
    predicted_covariance: torch.Tensor, target_covariance: torch.Tensor
) -> torch.Tensor:
    """
    Compute the rotation and scaling to transform
    each predicted_covariance into target_covariance.

    Args:
    predicted_covariance: torch.Tensor of shape (B, 6)
    target_covariance: torch.Tensor of shape (B, 6)

    Returns:
    torch.Tensor: Tensor of shape (B, 9) representing the scaling factors and lower rotation matrix.
    """
    B = predicted_covariance.shape[0]

    full_predicted_covariance = unpack_lower_triangular_to_full(predicted_covariance) + \
        torch.eye(3, device=predicted_covariance.device).unsqueeze(0) * 1e-3
    full_target_covariance = unpack_lower_triangular_to_full(target_covariance) + \
        torch.eye(3, device=predicted_covariance.device).unsqueeze(0) * 1e-3
    
    # Perform SVD on each batch of covariance matrices
    # U1, S1, V1 = torch.svd(full_predicted_covariance)
    # U2, S2, V2 = torch.svd(full_target_covariance)
    pred_eigvals, pred_eigvecs = torch.linalg.eigh(full_predicted_covariance)
    target_eigvals, target_eigvecs = torch.linalg.eigh(full_target_covariance)

    # res1 = torch.matmul(pred_eigvecs, torch.matmul(torch.diag_embed(pred_eigvals), pred_eigvecs.transpose(-2, -1)))
    # res2 = torch.matmul(target_eigvecs, torch.matmul(torch.diag_embed(target_eigvals), target_eigvecs.transpose(-2, -1)))

    S = target_eigvals / (pred_eigvals)
    R = torch.matmul(target_eigvecs, pred_eigvecs.transpose(-2, -1))
    Q = rotation_matrix_to_quaternion(R)
    return S, Q

def apply_rotation_and_scaling(predicted_covariance: torch.Tensor, S, Q):
    """
    Apply the rotation and scaling factors to predicted_covariance to calculate 
    target_covariance.
    
    Parameters:
    predicted_covariance (torch.Tensor): Batch of 3x3 covariance matrices of shape (B, 6)
    Q (torch.Tensor): Quaternions of shape (B, 4)
    S (torch.Tensor): Scaling factors of shape (B, 3)
    
    Returns:
    torch.Tensor: Transformed target_covariance of shape (B, 6)
    """

    pred_eigvals, pred_eigvecs = torch.linalg.eigh(
        unpack_lower_triangular_to_full(predicted_covariance) + \
            torch.eye(3, device=predicted_covariance.device).unsqueeze(0) * 1e-3
    )
    R = build_rotation(Q)
    
    # Apply scaling factors
    result = torch.matmul(
        R @ pred_eigvecs, 
        torch.matmul(
            torch.diag_embed(pred_eigvals * S), 
            pred_eigvecs.transpose(-2, -1) @ R.transpose(-2, -1)
        )
    ) - torch.eye(3, device=predicted_covariance.device).unsqueeze(0) * 1e-3
    return pack_full_to_lower_triangular(result)

def apply_cholesky(covariance: torch.Tensor) -> torch.Tensor:
    """Perform Cholesky decomposition on covariance (B x 6) to get lower triangular"""
    full_covariance = unpack_lower_triangular_to_full(covariance, symmetric=False) + \
        torch.eye(3, device=covariance.device).unsqueeze(0) * 1e-3
    L = torch.linalg.cholesky(full_covariance)
    return pack_full_to_lower_triangular(L)

def apply_inverse_cholesky(L: torch.Tensor) -> torch.Tensor:
    """Reconstruct full covariance matrix from lower triangular"""
    L_matrix = unpack_lower_triangular_to_full(L, symmetric=False)
    full_cov_matrix = L_matrix @ L_matrix.mT - torch.eye(3, device=L.device).unsqueeze(0) * 1e-3
    return pack_full_to_lower_triangular(full_cov_matrix)

def transform_covariance_matrix(
    scaling_rotation: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    """
    Apply the scaling and rotation to the covariance matrices.

    Args:
        scaling_rotation: torch.Tensor of shape (B, 7) representing the scaling factors and quaternions.
        covariance: torch.Tensor of shape (B, 6) representing the covariance matrices.

    Returns:
        torch.Tensor: Tensor of shape (B, 6) representing the transformed covariance matrices.
    """

    scaling = scaling_rotation[:, :3]
    rotation = scaling_rotation[:, 3:]

    full_covariance = unpack_lower_triangular_to_full(covariance)
    full_covariance = (
        torch.sqrt(scaling[:, None]) * full_covariance * torch.sqrt(scaling[:, :, None])
    )
    rotation_matrix = build_rotation(rotation)
    transformed_covariance = rotation_matrix @ full_covariance @ rotation_matrix.transpose(1, 2)

    return pack_full_to_lower_triangular(transformed_covariance)
