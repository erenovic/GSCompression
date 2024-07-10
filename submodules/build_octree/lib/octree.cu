#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#include <vector_types.h>
#include <device_launch_parameters.h>

#include "octree.h"

#define EPSILON 1e-8f

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    throw std::runtime_error(cudaGetErrorString(err)); \
}
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

__global__ void recursive_octree_kernel(
    const float* points, 
    int8_t* point_level_octant,
    int64_t* point_node_assignment,
    float* point_bbox_corners,
    const float* aabb_min, 
    const float* aabb_max,
    const int num_points, 
    const int max_depth) 
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    float current_min[3] = {aabb_min[0], aabb_min[1], aabb_min[2]};
    float current_max[3] = {aabb_max[0], aabb_max[1], aabb_max[2]};
    float point[3] = {points[point_idx * 3], points[point_idx * 3 + 1], points[point_idx * 3 + 2]};

    int64_t node_idx = 0;
    for (int depth = 0; depth < max_depth; ++depth) {
        float mid[3];
        int octant = 0;
        for (int i = 0; i < 3; ++i) {
            mid[i] = (current_min[i] + current_max[i]) * 0.5f;
            if (point[i] >= mid[i]) {
                octant |= (1 << i);
                current_min[i] = mid[i];
            } else {
                current_max[i] = mid[i];
            }
        }
        node_idx = node_idx * 8 + octant;

        point_level_octant[point_idx * max_depth + depth] = octant;
        point_node_assignment[point_idx * max_depth + depth] = node_idx;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
Octree::generate_octree(
    const torch::Tensor& points, 
    const torch::Tensor& aabb_min,
    const torch::Tensor& aabb_max,
    const int max_depth
) {
    CHECK_CUDA(points);
    CHECK_CUDA(aabb_min);
    CHECK_CUDA(aabb_max);
    
    int num_points = points.size(0);
    auto options_8 = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA, 0);
    torch::Tensor point_level_octants = torch::empty({num_points, max_depth}, options_8);

    auto options_64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
    torch::Tensor point_node_assignment = torch::empty({num_points, max_depth}, options_64);

    auto options_32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    torch::Tensor point_bbox_corners = torch::empty({num_points, 6}, options_32);

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    recursive_octree_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(), 
        point_level_octants.contiguous().data_ptr<int8_t>(),
        point_node_assignment.contiguous().data_ptr<int64_t>(),
        point_bbox_corners.contiguous().data_ptr<float>(),
        aabb_min.data_ptr<float>(),
        aabb_max.data_ptr<float>(),
        num_points, max_depth);

    cudaDeviceSynchronize();

    return std::make_tuple(point_level_octants, point_node_assignment, point_bbox_corners);
}


Octree::Octree(int max_depth) {
    this->max_depth = max_depth;
}

torch::Tensor Octree::get_point_node_assignment(
    const torch::Tensor points, 
    const torch::Tensor aabb_min, 
    const torch::Tensor aabb_max
) {
    auto result = generate_octree(points, aabb_min, aabb_max, max_depth);
    torch::Tensor point_node_assignment = std::get<1>(result);
    return point_node_assignment;
}

Octree::~Octree() {}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to calculate the determinant of a 3x3 matrix
__device__ float determinant3x3(float matrix[6]) {
    return matrix[0] * (matrix[3] * matrix[5] - matrix[4] * matrix[4]) -
           matrix[1] * (matrix[1] * matrix[5] - matrix[4] * matrix[2]) +
           matrix[2] * (matrix[1] * matrix[4] - matrix[3] * matrix[2]);
}

// Function to calculate the weight using the covariance matrix
__device__ float calculate_weight(float covariance[6], float opacity) {
    // Calculate the determinant of the covariance matrix

    float determinant = determinant3x3(covariance);

    determinant = fabsf(determinant) + EPSILON;

    // Calculate the square root of the determinant
    float characteristic_length = cbrtf(determinant);

    // Use a proportional factor to approximate the surface area
    // The proportional factor here is arbitrary, chosen to match typical scaling
    const float proportional_factor = 4.836; // This factor is to adjust the scale, it's not exact
    float surface_area_approx = proportional_factor * characteristic_length;

    return opacity * surface_area_approx;
}

__global__ void update_means_opacity_sh(
    const float* weights, 
    const float* mu, 
    const float* opacity,
    const float* sh,
    const int* node_ids, 
    float* new_mu,
    float* new_node_total_weight,
    float* new_opacity,
    float* new_sh,
    const int num_gaussians,
    const int num_sh_coeffs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    int node_id = node_ids[idx];
    float weight = weights[idx];

    // Weighted mu for current Gaussian
    float mu_x = mu[idx * 3 + 0];
    float mu_y = mu[idx * 3 + 1];
    float mu_z = mu[idx * 3 + 2];

    atomicAdd(&new_mu[node_id * 3 + 0], weight * mu_x);
    atomicAdd(&new_mu[node_id * 3 + 1], weight * mu_y);
    atomicAdd(&new_mu[node_id * 3 + 2], weight * mu_z);

    // Weighted total weight for current node
    atomicAdd(&new_node_total_weight[node_id], weight);

    // Weighted opacity for current Gaussian
    float op = opacity[idx];
    atomicAdd(&new_opacity[node_id], weight * op);

    // Weighted shs for current Gaussian
    for (int k = 0; k < num_sh_coeffs; k++) {
        atomicAdd(
            &new_sh[node_id * num_sh_coeffs + k], 
            weight * sh[idx * num_sh_coeffs + k]
        );
    }
}

__global__ void divide_by_total_weight(
    float* new_mu,
    float* new_node_total_weight,
    float* new_opacity,
    float* new_sh,
    const int num_new_nodes,
    const int num_sh_coeffs
) {
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= num_new_nodes) return;

    float total_weight = new_node_total_weight[node_id] + 1e-8; // Add epsilon to prevent division by zero

    new_mu[node_id * 3 + 0] /= total_weight;
    new_mu[node_id * 3 + 1] /= total_weight;
    new_mu[node_id * 3 + 2] /= total_weight;

    new_opacity[node_id] /= total_weight;

    for (int k = 0; k < num_sh_coeffs; k++) {
        new_sh[node_id * num_sh_coeffs + k] /= total_weight;
    }
}

__global__ void update_covariances(
    const float* weights, 
    const float* mu, 
    const float* sigma, 
    const int* node_ids, 
    float* new_mu, 
    float* new_sigma,
    const int num_gaussians
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    int node_id = node_ids[idx];
    float weight = weights[idx];

    // Current mu and new_mu values
    float mu_x = mu[idx * 3 + 0];
    float mu_y = mu[idx * 3 + 1];
    float mu_z = mu[idx * 3 + 2];

    float new_mu_x = new_mu[node_id * 3 + 0];
    float new_mu_y = new_mu[node_id * 3 + 1];
    float new_mu_z = new_mu[node_id * 3 + 2];

    // Difference mu_i - new_mu
    float diff_x = mu_x - new_mu_x;
    float diff_y = mu_y - new_mu_y;
    float diff_z = mu_z - new_mu_z;

    // Sigma indices
    int sigma_base_idx = idx * 6;
    int new_sigma_base_idx = node_id * 6;

    // Update covariance matrix components
    atomicAdd(&new_sigma[new_sigma_base_idx + 0], weight * (sigma[sigma_base_idx + 0] + diff_x * diff_x)); // sxx
    atomicAdd(&new_sigma[new_sigma_base_idx + 1], weight * (sigma[sigma_base_idx + 1] + diff_x * diff_y)); // sxy
    atomicAdd(&new_sigma[new_sigma_base_idx + 2], weight * (sigma[sigma_base_idx + 2] + diff_x * diff_z)); // sxz
    atomicAdd(&new_sigma[new_sigma_base_idx + 3], weight * (sigma[sigma_base_idx + 3] + diff_y * diff_y)); // syy
    atomicAdd(&new_sigma[new_sigma_base_idx + 4], weight * (sigma[sigma_base_idx + 4] + diff_y * diff_z)); // syz
    atomicAdd(&new_sigma[new_sigma_base_idx + 5], weight * (sigma[sigma_base_idx + 5] + diff_z * diff_z)); // szz
    // atomicAdd(&new_sigma[new_sigma_base_idx + 0], weight * (sigma[sigma_base_idx + 0])); // sxx
    // atomicAdd(&new_sigma[new_sigma_base_idx + 1], weight * (sigma[sigma_base_idx + 1])); // sxy
    // atomicAdd(&new_sigma[new_sigma_base_idx + 2], weight * (sigma[sigma_base_idx + 2])); // sxz
    // atomicAdd(&new_sigma[new_sigma_base_idx + 3], weight * (sigma[sigma_base_idx + 3])); // syy
    // atomicAdd(&new_sigma[new_sigma_base_idx + 4], weight * (sigma[sigma_base_idx + 4])); // syz
    // atomicAdd(&new_sigma[new_sigma_base_idx + 5], weight * (sigma[sigma_base_idx + 5])); // szz
}


__global__ void divide_sigma_by_total_weight(
    float* new_node_total_weight,
    float* new_sigma,
    const int num_new_nodes
) {
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= num_new_nodes) return;

    float total_weight = new_node_total_weight[node_id];

    int sigma_base_idx = node_id * 6;

    new_sigma[sigma_base_idx + 0] /= total_weight;
    new_sigma[sigma_base_idx + 1] /= total_weight;
    new_sigma[sigma_base_idx + 2] /= total_weight;
    new_sigma[sigma_base_idx + 3] /= total_weight;
    new_sigma[sigma_base_idx + 4] /= total_weight;
    new_sigma[sigma_base_idx + 5] /= total_weight;
}

// Host function to launch kernels
// Returns a tuple of tensors containing the updated means, opacities, spherical harmonics, and covariances
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
aggregate_gaussians_fwd(
    const torch::Tensor weights, 
    const torch::Tensor mu, 
    const torch::Tensor sigma,
    const torch::Tensor opacity,
    const torch::Tensor sh,
    const torch::Tensor node_ids, 
    const int num_new_nodes
) {

    CHECK_INPUT(weights);
    CHECK_INPUT(mu);
    CHECK_INPUT(opacity);
    CHECK_INPUT(sh);
    CHECK_INPUT(sigma);
    CHECK_INPUT(node_ids);

    // Assuming mu, sigma, and opacity have the same first dimension size as num_nodes
    // and sh has a second dimension for spherical harmonics coefficients.
    auto options = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());

    int num_sh_coeffs = sh.size(1);

    // Initialize new tensors inside the function
    torch::Tensor new_mu = torch::zeros({num_new_nodes, mu.size(1)}, options);
    torch::Tensor new_sigma = torch::zeros({num_new_nodes, sigma.size(1)}, options);
    torch::Tensor new_node_total_weight = torch::zeros({num_new_nodes}, options);
    torch::Tensor new_opacity = torch::zeros({num_new_nodes}, options);
    torch::Tensor new_sh = torch::zeros({num_new_nodes, num_sh_coeffs}, options);

    int num_gaussians = mu.size(0);

    // Kernel configurations
    int num_threads = 256;
    int num_gaussian_blocks = (num_gaussians + num_threads - 1) / num_threads;

    // Launch kernel to update means
    update_means_opacity_sh<<<num_gaussian_blocks, num_threads>>>(
        weights.data_ptr<float>(), 
        mu.data_ptr<float>(), 
        opacity.data_ptr<float>(),
        sh.data_ptr<float>(),
        node_ids.data_ptr<int>(), 
        new_mu.data_ptr<float>(),
        new_node_total_weight.data_ptr<float>(),
        new_opacity.data_ptr<float>(),
        new_sh.data_ptr<float>(),
        num_gaussians,
        num_sh_coeffs
    );

    // Ensure all updates to new_mu are complete
    cudaDeviceSynchronize();

    int num_node_blocks = (num_new_nodes + num_threads - 1) / num_threads;
    // Divide by the total weight to normalize the weights
    divide_by_total_weight<<<num_node_blocks, num_threads>>>(
        new_mu.data_ptr<float>(), 
        new_node_total_weight.data_ptr<float>(), 
        new_opacity.data_ptr<float>(),
        new_sh.data_ptr<float>(),
        num_new_nodes,
        num_sh_coeffs
    );

    // Ensure all updates to new_mu are complete
    cudaDeviceSynchronize();

    // Launch kernel to update covariances
    update_covariances<<<num_gaussian_blocks, num_threads>>>(
        weights.data_ptr<float>(), 
        mu.data_ptr<float>(), 
        sigma.data_ptr<float>(), 
        node_ids.data_ptr<int>(), 
        new_mu.data_ptr<float>(), 
        new_sigma.data_ptr<float>(),
        num_gaussians
    );

    // Ensure all updates to new_sigma are complete
    cudaDeviceSynchronize();

    // Divide the covariance by total weight to normalize the weights
    divide_sigma_by_total_weight<<<num_node_blocks, num_threads>>>(
        new_node_total_weight.data_ptr<float>(), 
        new_sigma.data_ptr<float>(),
        num_new_nodes
    );

    // return std::make_tuple(new_mu, new_sigma, new_opacity, new_sh, new_node_total_weight);
    return std::make_tuple(new_mu, new_sigma, new_opacity, new_sh, new_node_total_weight);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Kernel to calculate gradients for mu, opacity, and sh
__global__ void backward_update_gradients(
    const float* dL_dnew_mu, 
    const float* dL_dnew_opacity, 
    const float* dL_dnew_sh,
    const float* new_node_total_weight,
    const float* weights, 
    const float* mu, 
    const float* opacity, 
    const float* sh,
    const int* node_ids, 
    float* dL_dweights, 
    float* dL_dmu, 
    float* dL_dopacity, 
    float* dL_dsh,
    const int num_gaussians, 
    const int num_sh_coeffs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    int node_id = node_ids[idx];
    float weight = weights[idx];
    float total_weight = new_node_total_weight[node_id] + 1e-8; // Add epsilon to prevent division by zero

    for (int i = 0; i < 3; ++i) {
        float mu_grad = dL_dnew_mu[node_id * 3 + i] * weight / total_weight;
        atomicAdd(&dL_dmu[idx * 3 + i], mu_grad);
    }

    float opacity_grad = dL_dnew_opacity[node_id] * weight / total_weight;
    atomicAdd(&dL_dopacity[idx], opacity_grad);

    for (int k = 0; k < num_sh_coeffs; ++k) {
        float sh_grad = dL_dnew_sh[node_id * num_sh_coeffs + k] * weight / total_weight;
        atomicAdd(&dL_dsh[idx * num_sh_coeffs + k], sh_grad);
    }
    
    float weights_grad = 0;
    float weight_multiplier = (total_weight - weight) / (total_weight * total_weight + 1e-8);
    for (int i = 0; i < 3; ++i) {
        weights_grad += dL_dnew_mu[node_id * 3 + i] * mu[idx * 3 + i] * weight_multiplier;
    }

    // for (int i = 0; i < 6; ++i) {
    //     weights_grad += dL_dnew_sigma[node_id * 3 + i] * sigma[idx * 3 + i] * weight_multiplier;
    // }

    weights_grad += dL_dnew_opacity[node_id] * opacity[idx] * weight_multiplier;
    for (int k = 0; k < num_sh_coeffs; ++k) {
        weights_grad += dL_dnew_sh[node_id * num_sh_coeffs + k] * sh[idx * num_sh_coeffs + k] * weight_multiplier;
    }
    atomicAdd(&dL_dweights[idx], weights_grad);
}


// Kernel to calculate gradients for sigma
__global__ void backward_update_sigma_gradients(
    const float* dL_dnew_sigma, 
    const float* new_node_total_weight,
    const float* weights, 
    const float* mu, 
    const float* new_mu,
    const int* node_ids, 
    float* dL_dsigma,
    float* dL_dmu,
    const int num_gaussians
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    int node_id = node_ids[idx];
    float weight = weights[idx];
    float total_weight = new_node_total_weight[node_id];

    // Extract the differences mu_i - new_mu for each component
    float diff_x = mu[idx * 3 + 0] - new_mu[node_id * 3 + 0];
    float diff_y = mu[idx * 3 + 1] - new_mu[node_id * 3 + 1];
    float diff_z = mu[idx * 3 + 2] - new_mu[node_id * 3 + 2];

    // Base indices for sigma in the flattened array
    int sigma_base_idx = idx * 6; // input sigma
    int new_sigma_base_idx = node_id * 6; // gradients w.r.t. new_sigma

    float normalized_weight = weight / total_weight;

    // Calculate gradients for each component of sigma
    atomicAdd(&dL_dsigma[sigma_base_idx + 0], dL_dnew_sigma[new_sigma_base_idx + 0] * normalized_weight);
    atomicAdd(&dL_dsigma[sigma_base_idx + 1], dL_dnew_sigma[new_sigma_base_idx + 1] * normalized_weight);
    atomicAdd(&dL_dsigma[sigma_base_idx + 2], dL_dnew_sigma[new_sigma_base_idx + 2] * normalized_weight);
    atomicAdd(&dL_dsigma[sigma_base_idx + 3], dL_dnew_sigma[new_sigma_base_idx + 3] * normalized_weight);
    atomicAdd(&dL_dsigma[sigma_base_idx + 4], dL_dnew_sigma[new_sigma_base_idx + 4] * normalized_weight);
    atomicAdd(&dL_dsigma[sigma_base_idx + 5], dL_dnew_sigma[new_sigma_base_idx + 5] * normalized_weight);

    // Calculate gradients for mu
    float dSigma0_dmu0 = 2 * normalized_weight * diff_x * (1 - normalized_weight);
    float dSigma1_dmu0 = normalized_weight * diff_y * (1 - normalized_weight);
    float dSigma2_dmu0 = normalized_weight * diff_z * (1 - normalized_weight);
    atomicAdd(
        &dL_dmu[idx * 3 + 0], 
        dL_dnew_sigma[new_sigma_base_idx + 0] * dSigma0_dmu0 + dL_dnew_sigma[new_sigma_base_idx + 1] * dSigma1_dmu0 + dL_dnew_sigma[new_sigma_base_idx + 2] * dSigma2_dmu0
    );

    float dSigma1_dmu1 = normalized_weight * diff_x * (1 - normalized_weight);
    float dSigma3_dmu1 = 2 * normalized_weight * diff_y * (1 - normalized_weight);
    float dSigma4_dmu1 = normalized_weight * diff_z * (1 - normalized_weight);
    atomicAdd(
        &dL_dmu[idx * 3 + 1], 
        dL_dnew_sigma[new_sigma_base_idx + 1] * dSigma1_dmu1 + dL_dnew_sigma[new_sigma_base_idx + 3] * dSigma3_dmu1 + dL_dnew_sigma[new_sigma_base_idx + 4] * dSigma4_dmu1
    );

    float dSigma2_dmu2 = normalized_weight * diff_x * (1 - normalized_weight);
    float dSigma4_dmu2 = normalized_weight * diff_y * (1 - normalized_weight);
    float dSigma5_dmu2 = 2 * normalized_weight * diff_z * (1 - normalized_weight);
    atomicAdd(
        &dL_dmu[idx * 3 + 2], 
        dL_dnew_sigma[new_sigma_base_idx + 2] * dSigma2_dmu2 + dL_dnew_sigma[new_sigma_base_idx + 4] * dSigma4_dmu2 + dL_dnew_sigma[new_sigma_base_idx + 5] * dSigma5_dmu2
    );
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
aggregate_gaussians_bwd(
    const torch::Tensor dL_dnew_mu,
    const torch::Tensor dL_dnew_sigma, 
    const torch::Tensor dL_dnew_opacity, 
    const torch::Tensor dL_dnew_sh, 
    const torch::Tensor new_node_total_weight,
    const torch::Tensor weights, 
    const torch::Tensor mu,
    const torch::Tensor sigma, 
    const torch::Tensor opacity,
    const torch::Tensor sh, 
    const torch::Tensor new_mu,
    const torch::Tensor new_sigma,
    const torch::Tensor node_ids,
    const int num_gaussians, 
    const int num_new_nodes
) {
    CHECK_INPUT(dL_dnew_mu);
    CHECK_INPUT(dL_dnew_sigma);
    CHECK_INPUT(dL_dnew_opacity);
    CHECK_INPUT(dL_dnew_sh);
    CHECK_INPUT(new_node_total_weight);
    CHECK_INPUT(weights);
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    CHECK_INPUT(opacity);
    CHECK_INPUT(sh);
    CHECK_INPUT(new_mu);
    CHECK_INPUT(new_sigma);
    CHECK_INPUT(node_ids);

    auto options = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());
    int num_sh_coeffs = sh.size(1);

    torch::Tensor dL_dweights = torch::zeros_like(weights, options);
    torch::Tensor dL_dmu = torch::zeros_like(mu, options);
    torch::Tensor dL_dsigma = torch::zeros_like(sigma, options);
    torch::Tensor dL_dopacity = torch::zeros_like(opacity, options);
    torch::Tensor dL_dsh = torch::zeros_like(sh, options);
    
    int num_threads = 256;
    int num_blocks = (num_gaussians + num_threads - 1) / num_threads;

    // Launch kernels
    backward_update_gradients<<<num_blocks, num_threads>>>(
        dL_dnew_mu.data_ptr<float>(), 
        dL_dnew_opacity.data_ptr<float>(), 
        dL_dnew_sh.data_ptr<float>(),
        new_node_total_weight.data_ptr<float>(),
        weights.data_ptr<float>(), 
        mu.data_ptr<float>(), 
        opacity.data_ptr<float>(), 
        sh.data_ptr<float>(),
        node_ids.data_ptr<int>(), 
        dL_dweights.data_ptr<float>(), 
        dL_dmu.data_ptr<float>(), 
        dL_dopacity.data_ptr<float>(), 
        dL_dsh.data_ptr<float>(),
        num_gaussians, num_sh_coeffs
    );

    cudaDeviceSynchronize();

    backward_update_sigma_gradients<<<num_blocks, num_threads>>>(
        dL_dnew_sigma.data_ptr<float>(), 
        new_node_total_weight.data_ptr<float>(),
        weights.data_ptr<float>(), 
        mu.data_ptr<float>(), 
        new_mu.data_ptr<float>(),
        node_ids.data_ptr<int>(), 
        dL_dsigma.data_ptr<float>(),
        dL_dmu.data_ptr<float>(), 
        num_gaussians
    );

    cudaDeviceSynchronize();

    return std::make_tuple(dL_dweights, dL_dmu, dL_dsigma, dL_dopacity, dL_dsh);
}

