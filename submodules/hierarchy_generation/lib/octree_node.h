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

    // float features[48]; // Features of the node
    float opacity; // Opacity of the node
    float scaling[3]; // Scaling of the node
    float rotation[4]; // Rotation of the node
    float features[48]; // Features of the node
    float total_weight; // Total weight of the node
    float weight; // Weight of the node

    __device__ __host__ OctreeNode();
    __device__ __host__ OctreeNode(
        float* bbox_min, float* bbox_max, 
        int depth, OctreeNode* parent);
    
    // __device__ __host__ ~OctreeNode(); // Destructor for cleanup

    // __host__ torch::Tensor get_node_position();

    // __device__ __host__ int getChildIndex(const float pos[3]) const;

    __device__ __host__ std::vector<int> get_children_indices() const {
        std::vector<int> indices;
        for (int i = 0; i < 8; i++) {
            if (children[i] != nullptr) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    std::vector<OctreeNode> get_children() const;

    __device__ __host__ bool isLeaf() const {
        return (
            (children[0] == nullptr) && 
            (children[1] == nullptr) && 
            (children[2] == nullptr) && 
            (children[3] == nullptr) && 
            (children[4] == nullptr) && 
            (children[5] == nullptr) && 
            (children[6] == nullptr) && 
            (children[7] == nullptr)
        );
    }

    OctreeNode get_parent() const;

    // Getter methods
    torch::Tensor get_opacity() const;
    torch::Tensor get_scaling() const;
    torch::Tensor get_rotation() const;
    torch::Tensor get_features() const;
    torch::Tensor get_weight() const;
};