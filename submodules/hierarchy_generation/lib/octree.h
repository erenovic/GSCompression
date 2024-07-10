#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <array>


struct BoundingBox {
    std::array<float, 3> min;
    std::array<float, 3> max;
    std::array<float, 3> size;
    std::array<float, 3> center;

    __device__ __host__ BoundingBox() {
        for (int i = 0; i < 3; i++) {
            min[i] = 0.0f;
            max[i] = 0.0f;
            size[i] = 0.0f;
            center[i] = 0.0f;
        }
    }

    __device__ __host__ BoundingBox(const float bbox_min[3], const float bbox_max[3]) {
        for (int i = 0; i < 3; i++) {
            min[i] = bbox_min[i];
            max[i] = bbox_max[i];
            size[i] = max[i] - min[i];
            center[i] = (min[i] + max[i]) / 2.0f;
        }
    }

    __device__ __host__ bool inside(const float point[3]) const {
        return (
            (point[0] >= min[0]) && (point[0] <= max[0]) &&
            (point[1] >= min[1]) && (point[1] <= max[1]) &&
            (point[2] >= min[2]) && (point[2] <= max[2])
        );
    }
};


class OctreeNode {
public:
    // std::array<float, 3> position;  // Position of the node
    BoundingBox bbox;  // Bounding box of the node
    int depth;  // Depth of the node in the octree
    
    OctreeNode* parent; // Pointer to parent node
    std::array<OctreeNode*, 8> children; // Pointers to child nodes

    // // float features[48]; // Features of the node
    // float opacity; // Opacity of the node
    // float scaling[3]; // Scaling of the node
    // float rotation[4]; // Rotation of the node
    // float features[48]; // Features of the node
    // float total_weight; // Total weight of the node
    // float weight; // Weight of the node

    // // Gradients
    // float dL_dposition[3]; // Gradient of loss w.r.t. position
    // float dL_dopacity; // Gradient of loss w.r.t. opacity
    // float dL_dscaling[3]; // Gradient of loss w.r.t. scaling
    // float dL_drotation[4]; // Gradient of loss w.r.t. rotation
    // float dL_dfeatures[48]; // Gradient of loss w.r.t. features
    // float dL_dweight; // Gradient of loss w.r.t. weight

    __device__ __host__ OctreeNode();
    __device__ __host__ OctreeNode(
        float* bbox_min, float* bbox_max, 
        int depth, OctreeNode* parent);
    
    // __device__ __host__ ~OctreeNode(); // Destructor for cleanup

    // __host__ torch::Tensor get_node_position();

    // __device__ __host__ int getChildIndex(const float pos[3]) const;

    // __device__ __host__ std::vector<int> get_children_indices() const {
    //     std::vector<int> indices;
    //     for (int i = 0; i < 8; i++) {
    //         if (children[i] != nullptr) {
    //             indices.push_back(i);
    //         }
    //     }
    //     return indices;
    // }

    // std::vector<OctreeNode> get_children() const;

    // __device__ __host__ bool isLeaf() const {
    //     return (
    //         (children[0] == nullptr) && 
    //         (children[1] == nullptr) && 
    //         (children[2] == nullptr) && 
    //         (children[3] == nullptr) && 
    //         (children[4] == nullptr) && 
    //         (children[5] == nullptr) && 
    //         (children[6] == nullptr) && 
    //         (children[7] == nullptr)
    //     );
    // }

    // OctreeNode get_parent() const;

    // // Getter methods
    // torch::Tensor get_opacity() const;
    // torch::Tensor get_scaling() const;
    // torch::Tensor get_rotation() const;
    // torch::Tensor get_features() const;
    // torch::Tensor get_weight() const;
};

class Octree {
public:
    int max_depth;  // Maximum depth of the octree
    OctreeNode* device_nodes;   // Array of nodes

    Octree(
        const torch::Tensor points, 
        const torch::Tensor aabb_min, 
        const torch::Tensor aabb_max,
        int max_depth);
    
    ~Octree();

    OctreeNode get_node_by_id(int node_id) const;

    std::vector<int> get_node_per_depth_count() const;
    int get_node_count() const;

    // std::vector<OctreeNode> get_nodes_at_depth(int depth) const;

    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    torch::Tensor select_tree_cut(
        const torch::Tensor projection_matrix, 
        const torch::Tensor view_matrix, 
        const int H, const int W,
        const float granularity_threshold,
        const int chosen_depth) const;

    void build(
        const torch::Tensor points, 
        const torch::Tensor aabb_min, 
        const torch::Tensor aabb_max);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    set_node_attributes(
        const torch::Tensor g_positions,
        const torch::Tensor g_covariances,
        const torch::Tensor g_features,
        const torch::Tensor g_opacities,
        const torch::Tensor g_weights,
        const torch::Tensor filtered_node_indices);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    set_node_attributes_backward(
        const torch::Tensor dL_dnew_position,
        const torch::Tensor dL_dnew_covariances,
        const torch::Tensor dL_dnew_features,
        const torch::Tensor dL_dnew_opacities,
        const torch::Tensor node_indices,
        const torch::Tensor point_weights,
        const torch::Tensor weights,
        const torch::Tensor total_weights) const;

    torch::Tensor get_nodes_at_depth(const int depth) const;

    torch::Tensor get_nodes_w_max_granularity(
    const torch::Tensor projection_matrix, 
    const torch::Tensor view_matrix, 
    const int H, const int W,
    const float granularity_threshold) const;

private:
    OctreeNode** point_to_node_pointer; // Array of pointers to nodes for each point

    int num_points;  // Number of points
    int* device_node_count;    // Number of nodes
    int* device_node_per_depth_count; // Number of nodes per depth

    // torch::Tensor filter_nodes_at_depth(int depth) const;
};