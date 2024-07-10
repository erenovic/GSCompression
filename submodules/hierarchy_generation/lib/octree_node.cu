#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "octree_node.h"


__device__ __host__ OctreeNode::OctreeNode() {
    bbox = BoundingBox();
    depth = 0;
    parent = nullptr;

    for (int i = 0; i < 8; ++i) {
        children[i] = nullptr;
    }
    for (int i = 0; i < 3; ++i) {
        scaling[i] = 0.0f;
    }
    for (int i = 0; i < 4; ++i) {
        rotation[i] = 0.0f;
    }
    for (int i = 0; i < 48; ++i) {
        features[i] = 0.0f;
    }
    opacity = 0.0f;
    total_weight = 0.0f;
    weight = 0.0f;
}

__device__ __host__ OctreeNode::OctreeNode(
    float* bbox_min, float* bbox_max,
    int depth, OctreeNode* parent
) {
    bbox = BoundingBox(bbox_min, bbox_max);
    this->depth = depth;
    this->parent = parent;

    // Pointers to children will be updated in build_octree_kernel
    for (int i = 0; i < 8; ++i) {
        children[i] = nullptr;
    }
    // Features of the nodes will be updated in 
    // set_node_attributes_from_points_kernel and set_node_attributes_from_nodes_kernel
    for (int i = 0; i < 3; ++i) {
        scaling[i] = 0.0f;
    }
    for (int i = 0; i < 4; ++i) {
        rotation[i] = 0.0f;
    }
    for (int i = 0; i < 48; ++i) {
        features[i] = 0.0f;
    }
    opacity = 0.0f;
    total_weight = 0.0f;
    weight = 0.0f;
}

std::vector<OctreeNode> OctreeNode::get_children() const {
    std::vector<OctreeNode> childrens;
    for (int i = 0; i < 8; ++i) {
        if (children[i] != nullptr) {
            OctreeNode node;
            cudaMemcpy(&node, children[i], sizeof(OctreeNode), cudaMemcpyDeviceToHost);
            childrens.push_back(node);
        }
    }
    return childrens;
}

OctreeNode OctreeNode::get_parent() const {
    OctreeNode parent;
    cudaMemcpy(&parent, this->parent, sizeof(OctreeNode), cudaMemcpyDeviceToHost);
    return parent;
}

torch::Tensor OctreeNode::get_opacity() const {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor opacity_tensor = torch::empty({1}, options);
    opacity_tensor[0] = opacity;
    return opacity_tensor;
}

torch::Tensor OctreeNode::get_scaling() const {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor scaling_tensor = torch::empty({3}, options);
    for (int i = 0; i < 3; ++i) {
       scaling_tensor[i] = scaling[i];
    }
    return scaling_tensor;
}

torch::Tensor OctreeNode::get_rotation() const {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor rotation_tensor = torch::empty({4}, options);
    for (int i = 0; i < 4; ++i) {
       rotation_tensor[i] = rotation[i];
    }
    return rotation_tensor;
}

torch::Tensor OctreeNode::get_features() const {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor features_tensor = torch::empty({48}, options);
    for (int i = 0; i < 48; ++i) {
       features_tensor[i] = features[i];
    }
    return features_tensor;
}

torch::Tensor OctreeNode::get_weight() const {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor weight_tensor = torch::empty({1}, options);
    weight_tensor[0] = weight;
    return weight_tensor;
}