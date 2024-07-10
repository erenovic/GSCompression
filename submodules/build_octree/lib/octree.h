/*
 * Eren Cetin
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

class Octree {
public:
    int max_depth;
    torch::Tensor point_node_assignment;
    // torch::Tensor point_level_octants;

    Octree(int max_depth);
    
    ~Octree();

    torch::Tensor get_point_node_assignment(
        const torch::Tensor points, 
        const torch::Tensor aabb_min, 
        const torch::Tensor aabb_max);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    generate_octree(
        const torch::Tensor& points, 
        const torch::Tensor& aabb_min,
        const torch::Tensor& aabb_max,
        const int max_depth);
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
aggregate_gaussians_fwd(
    const torch::Tensor weights, 
    const torch::Tensor mu, 
    const torch::Tensor sigma,
    const torch::Tensor opacity,
    const torch::Tensor sh,
    const torch::Tensor node_ids, 
    const int num_new_nodes);

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
    const int num_new_nodes);
