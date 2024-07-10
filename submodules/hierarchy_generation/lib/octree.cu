#include "octree.h"
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    throw std::runtime_error(cudaGetErrorString(err)); \
}
#define EPSILON 1e-8f
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }


// CUDA Kernel to get minimum node depth among filtered node indices
__global__ void get_min_node_depth_kernel(
    OctreeNode* nodes,
    int* filtered_node_indices,
    int* min_depth,
    int filtered_node_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= filtered_node_count) return;

    int node_idx = filtered_node_indices[idx];
    OctreeNode* node = &nodes[node_idx];

    atomicMin(min_depth, node->depth);
}

// CUDA Kernel to get node indices for nodes at a given depth
__global__ void get_nodes_at_depth_kernel(
    OctreeNode* nodes,
    OctreeNode** point_to_node_pointer,
    int* node_indices,
    int num_points,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    OctreeNode* node = point_to_node_pointer[idx];

    while (node->depth != depth) {
        node = node->parent;
    }

    node_indices[idx] = node - nodes;
}

// CUDA Kernel to propagate gradients from new node attributes to children
__global__ void propagate_output_node_gradients_kernel(
    OctreeNode* nodes,
    const float* dL_dnew_position,
    const float* dL_dnew_covariances,
    const float* dL_dnew_features,
    const float* dL_dnew_opacities,
    float* dL_dposition,
    float* dL_dcovariances,
    float* dL_dfeatures,
    float* dL_dopacities,
    const int* node_indices,
    int* min_depth_of_new_nodes,
    const int features_dim,
    int num_new_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_new_nodes) return;

    int node_idx = node_indices[idx];
    
    for (int i = 0; i < 3; ++i) {
        atomicAdd(&dL_dposition[node_idx * 3 + i], dL_dnew_position[idx * 3 + i]);
    }
    for (int i = 0; i < 6; ++i) {
        atomicAdd(&dL_dcovariances[node_idx * 6 + i], dL_dnew_covariances[idx * 6 + i]);
    }
    for (int i = 0; i < features_dim; ++i) {
        atomicAdd(&dL_dfeatures[node_idx * features_dim + i], dL_dnew_features[idx * features_dim + i]);
    }
    atomicAdd(&dL_dopacities[node_idx], dL_dnew_opacities[idx]);

    // Update min_depth_of_new_nodes
    atomicMin(min_depth_of_new_nodes, nodes[node_idx].depth);
}

__global__ void propagate_gradients_from_parent_to_child(
    OctreeNode* nodes, 
    float* dL_dposition,
    float* dL_dcovariances,
    float* dL_dfeatures,
    float* dL_dopacities,
    float* weights,
    float* total_weights,
    const int features_dim,
    int num_nodes, int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    OctreeNode* node = &nodes[idx];

    if (node->depth != depth) return;
    
    float total_weight = total_weights[node - nodes] + EPSILON;

    for (int i = 0; i < 8; ++i) {
        OctreeNode* child = node->children[i];
        if (child != nullptr) {
            float weight = weights[child - nodes];

            for (int j = 0; j < 3; ++j) {
                // child->dL_dposition[j] = node->dL_dposition[j] * (weight / total_weight);
                atomicAdd(
                    &dL_dposition[(child - nodes) * 3 + j], 
                    dL_dposition[(node - nodes) * 3 + j] * (weight / total_weight)
                );
            }
            for (int j = 0; j < 6; ++j) {
                // child->dL_dposition[j] = node->dL_dposition[j] * (weight / total_weight);
                atomicAdd(
                    &dL_dcovariances[(child - nodes) * 6 + j], 
                    dL_dcovariances[(node - nodes) * 6 + j] * (weight / total_weight)
                );
            }
            for (int j = 0; j < features_dim; ++j) {
                // child->dL_dfeatures[j] = node->dL_dfeatures[j] * (weight / total_weight);
                atomicAdd(
                    &dL_dfeatures[(child - nodes) * features_dim + j], 
                    dL_dfeatures[(node - nodes) * features_dim + j] * (weight / total_weight)
                );
            }
            atomicAdd(&dL_dopacities[child - nodes], dL_dopacities[node - nodes] * (weight / total_weight));
        }
    }
}

__global__ void propagate_gradients_from_leaf_node_to_points(
    OctreeNode* nodes,
    float* dL_dposition,
    float* dL_dcovariances,
    float* dL_dfeatures,
    float* dL_dopacities,
    float* out_dL_dposition,
    float* out_dL_dcovariances,
    float* out_dL_dfeatures,
    float* out_dL_dopacities,
    const float* point_weights,
    const float* total_weights,
    OctreeNode** point_to_node_pointer,
    const int features_dim,
    const int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    OctreeNode* node = point_to_node_pointer[idx];

    float weight = point_weights[idx];
    float total_weight = total_weights[node - nodes] + EPSILON;

    // Propagate gradients to points
    for (int i = 0; i < 3; ++i)  {
        // dL_dposition[idx * 3 + i] = node->dL_dposition[idx * 3 + i] * (weight / total_weight);
        atomicAdd(&out_dL_dposition[idx * 3 + i], dL_dposition[(node - nodes) * 3 + i] * (weight / total_weight));
    }
    for (int i = 0; i < 6; ++i)  {
        // dL_dposition[idx * 3 + i] = node->dL_dposition[idx * 3 + i] * (weight / total_weight);
        atomicAdd(&out_dL_dcovariances[idx * 6 + i], dL_dcovariances[(node - nodes) * 6 + i] * (weight / total_weight));
    }
    for (int i = 0; i < features_dim; ++i) {
        // node->dL_dfeatures[i], node->dL_dfeatures[i] * (weight / total_weight);
        atomicAdd(&out_dL_dfeatures[idx * features_dim + i], dL_dfeatures[(node - nodes) * features_dim + i] * (weight / total_weight));
    }
    atomicAdd(&out_dL_dopacities[idx], dL_dopacities[(node - nodes)] * (weight / total_weight));
}

// CUDA Kernel to get pointers to nodes
__global__ void get_node_indices_from_flags_kernel(
    int* filtered_node_flags,
    int* filtered_node_indices,
    int* filtered_node_count,
    int node_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;

    if (filtered_node_flags[idx] == 1) {
        // Placing the node index in the filtered node indices array densely
        int index = atomicSub(filtered_node_count, 1);
        filtered_node_indices[index-1] = idx;
    }
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix) {
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix) {
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

// CUDA Kernel to calculate 2D bounding box
__device__ void calculate_bounding_box(
    OctreeNode* node,
    float* bbox2D,
    const float* projection_matrix,
    const float* view_matrix,
    const int H, const int W
) {
    float bbox_min[3] = { node->bbox.min[0], node->bbox.min[1], node->bbox.min[2] };
    float bbox_max[3] = { node->bbox.max[0], node->bbox.max[1], node->bbox.max[2] };

    // Project bounding box vertices to 2D using projection_matrix
    float vertices[8][2];
    for (int i = 0; i < 8; ++i) {
        float x = (i & 1) ? bbox_max[0] : bbox_min[0];
        float y = (i & 2) ? bbox_max[1] : bbox_min[1];
        float z = (i & 4) ? bbox_max[2] : bbox_min[2];

        float3 p_orig = { x, y, z };

        // Bring points to screen space
        float4 p_hom = transformPoint4x4(p_orig, projection_matrix);
        float p_w = 1.0f / (p_hom.w + 0.0000001f);
        float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
        float3 p_view = transformPoint4x3(p_orig, view_matrix);

        vertices[i][0] = ndc2Pix(p_proj.x, W);
        vertices[i][1] = ndc2Pix(p_proj.y, H);
    }

    // Calculate bounding box in 2D
    float min_x = vertices[0][0], max_x = vertices[0][0];
    float min_y = vertices[0][1], max_y = vertices[0][1];
    for (int i = 1; i < 8; ++i) {
        if (vertices[i][0] < min_x) min_x = vertices[i][0];
        if (vertices[i][0] > max_x) max_x = vertices[i][0];
        if (vertices[i][1] < min_y) min_y = vertices[i][1];
        if (vertices[i][1] > max_y) max_y = vertices[i][1];
    }

    bbox2D[0] = min_x;
    bbox2D[1] = min_y;
    bbox2D[2] = max_x;
    bbox2D[3] = max_y;
}

// // CUDA Kernel to filter nodes
// __global__ void filter_nodes_for_granularity_kernel(
//     float* node_bounding_boxes,
//     int* selected_node_flags,
//     int* filtered_node_count,
//     int node_count,
//     float granularity_threshold
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= node_count) return;

//     float min_x = node_bounding_boxes[idx * 4 + 0];
//     float min_y = node_bounding_boxes[idx * 4 + 1];
//     float max_x = node_bounding_boxes[idx * 4 + 2];
//     float max_y = node_bounding_boxes[idx * 4 + 3];

//     float max_dim = max(max_x - min_x, max_y - min_y);

//     if (max_dim < granularity_threshold) {
//         // Mark this node as selected
//         selected_node_flags[idx] = 1;

//         // Add this node to filtered nodes
//         int index = atomicAdd(filtered_node_count, 1);
//     }
// }

// CUDA Kernel to filter nodes
__global__ void filter_nodes_for_granularity_kernel(
    OctreeNode* nodes,
    OctreeNode** point_to_node_pointer,
    const float* projection_matrix,
    const float* view_matrix,
    const int H, const int W,
    int* node_indices,
    int num_points,
    float granularity_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    OctreeNode* node = point_to_node_pointer[idx];

    // Calculate the bounding box projection
    float bbox2D[4];
    calculate_bounding_box(node, bbox2D, projection_matrix, view_matrix, H, W);

    // Check if the node satisfies the granularity threshold
    float min_x = bbox2D[0];
    float min_y = bbox2D[1];
    float max_x = bbox2D[2];
    float max_y = bbox2D[3];

    float max_dim = max(max_x - min_x, max_y - min_y);

    while (max_dim < granularity_threshold) {
        // Mark this node as selected
        node_indices[idx] = node - nodes;
        
        // Move to parent node
        node = node->parent;
        
        // Calculate the bounding box projection
        calculate_bounding_box(node, bbox2D, projection_matrix, view_matrix, H, W);

        // Check if the node satisfies the granularity threshold
        min_x = bbox2D[0];
        min_y = bbox2D[1];
        max_x = bbox2D[2];
        max_y = bbox2D[3];

        max_dim = max(max_x - min_x, max_y - min_y);

        if (node->depth == 0) break;
    }
}

// CUDA Kernel to filter nodes adhering the granularity threshold and 
// prune descendants of selected nodes at a specific depth
__global__ void filter_and_prune_nodes_for_granularity_kernel(
    OctreeNode* nodes,
    const float* projection_matrix,
    const float* view_matrix,
    const int H, const int W,
    int* filtered_node_flags,
    int* filtered_node_count,
    int node_count,
    float granularity_threshold,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;

    OctreeNode* node = &nodes[idx];
    
    // Only process nodes at the given depth
    if (node->depth != depth) return;

    // Calculate the bounding box projection
    float bbox2D[4];
    calculate_bounding_box(node, bbox2D, projection_matrix, view_matrix, H, W);

    // Check if the node satisfies the granularity threshold
    float min_x = bbox2D[0];
    float min_y = bbox2D[1];
    float max_x = bbox2D[2];
    float max_y = bbox2D[3];

    float max_dim = max(max_x - min_x, max_y - min_y);

    if (max_dim < granularity_threshold) {
        // Mark this node as selected
        atomicAdd(filtered_node_count, 1);
        filtered_node_flags[idx] = 1;

        // Prune descendants of this node
        for (int i = 0; i < 8; ++i) {
            OctreeNode* child = node->children[i];
            // Check if the child pointer is valid and selected
            if ((child != nullptr) && (filtered_node_flags[child - nodes] == 1)) {
                // Mark the child as not selected
                filtered_node_flags[child - nodes] = 0;

                // Subtract the child from the filtered node count
                atomicSub(filtered_node_count, 1);
            }
        }
    }
}

__global__ void count_nodes_at_all_depths(
    OctreeNode* nodes,
    int* node_per_depth_count,
    int node_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;

    OctreeNode* node = &nodes[idx];
    int depth = node->depth;
    atomicAdd(&node_per_depth_count[depth], 1);
}

__global__ void normalize_node_position_sh_opacity_at_depth_kernel(
    OctreeNode* nodes,
    float* node_centers,
    float* node_shs,
    float* node_opacity,
    float* node_total_weight,
    int features_dim,
    int node_count,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;

    OctreeNode* node = &nodes[idx];

    if (node->depth != depth) return;

    float node_weight = node_total_weight[node - nodes] + EPSILON;

    // Normalize bounding box center
    for (int i = 0; i < 3; ++i) {
        node_centers[3 * (node - nodes) + i] = node_centers[3 * (node - nodes) + i] / node_weight;
    }
    // Normalize SHs
    for (int i = 0; i < features_dim; ++i) {
        node_shs[features_dim * (node - nodes) + i] = node_shs[features_dim * (node - nodes) + i] / node_weight;
    }
    node_opacity[(node - nodes)] = node_opacity[(node - nodes)] / node_weight;
}

__global__ void normalize_node_attributes_at_depth_kernel(
    OctreeNode* nodes,
    float* node_centers,
    float* node_covariance,
    float* node_shs,
    float* node_opacity,
    float* node_total_weight,
    int features_dim,
    int node_count,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;

    OctreeNode* node = &nodes[idx];

    if (node->depth != depth) return;

    float node_weight = node_total_weight[node - nodes] + EPSILON;

    // Normalize bounding box center
    for (int i = 0; i < 3; ++i) {
        node_centers[3 * (node - nodes) + i] = node_centers[3 * (node - nodes) + i] / node_weight;
    }
    for (int i = 0; i < 6; ++i) {
        node_covariance[6 * (node - nodes) + i] = node_covariance[6 * (node - nodes) + i] / node_weight;
    }
    // Normalize SHs
    for (int i = 0; i < features_dim; ++i) {
        node_shs[features_dim * (node - nodes) + i] = node_shs[features_dim * (node - nodes) + i] / node_weight;
    }
    node_opacity[(node - nodes)] = node_opacity[(node - nodes)] / node_weight;
}

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

__global__ void set_leaf_node_position_sh_opacity_kernel(
    OctreeNode* nodes,
    const float* g_positions,
    const float* g_features,
    const float* g_opacities,
    const float* g_weights,
    float* node_centers,
    float* node_features,
    float* node_opacities,
    float* node_total_weight,
    OctreeNode** point_to_node_pointer,
    int features_dim,
    int num_points,
    int max_depth
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    OctreeNode* node = point_to_node_pointer[point_idx];

    if (node->depth != max_depth) return;

    for (int i = 0; i < 3; ++i) {
        atomicAdd(&node_centers[3 * (node - nodes) + i], g_weights[point_idx] * g_positions[point_idx * 3 + i]);
    }
    for (int i = 0; i < features_dim; ++i) {
        // atomicAdd(&node->features[i], weights[point_idx] * features[point_idx * 48 + i]);
        atomicAdd(&node_features[features_dim * (node - nodes) + i], g_weights[point_idx] * g_features[point_idx * features_dim + i]);
    }
    atomicAdd(&node_opacities[(node - nodes)], g_weights[point_idx] * g_opacities[point_idx]);

    // atomicAdd(&node->total_weight, weights[point_idx]);
    atomicAdd(&node_total_weight[node - nodes], g_weights[point_idx]);
}

__global__ void set_leaf_node_covariance_kernel(
    OctreeNode* nodes,
    const float* g_positions,
    const float* g_covariances,
    const float* g_weights,
    float* node_centers,
    float* node_covariances,
    float* total_weights,
    OctreeNode** point_to_node_pointer,
    int num_points,
    int max_depth
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    OctreeNode* node = point_to_node_pointer[point_idx];

    if (node->depth != max_depth) return;

    float node_total_weight = total_weights[node - nodes] + EPSILON;

    float position_difference[3] = {
        g_positions[3 * point_idx] - node_centers[3 * (node - nodes)],
        g_positions[3 * point_idx + 1] - node_centers[3 * (node - nodes) + 1],
        g_positions[3 * point_idx + 2] - node_centers[3 * (node - nodes) + 2]
    };

    float unweighted[6] = {
        g_covariances[6 * point_idx] + position_difference[0] * position_difference[0],
        g_covariances[6 * point_idx + 1] + position_difference[0] * position_difference[1],
        g_covariances[6 * point_idx + 2] + position_difference[0] * position_difference[2],
        g_covariances[6 * point_idx + 3] + position_difference[1] * position_difference[1],
        g_covariances[6 * point_idx + 4] + position_difference[1] * position_difference[2],
        g_covariances[6 * point_idx + 5] + position_difference[2] * position_difference[2]
    };

    for (int i = 0; i < 6; ++i) {
        atomicAdd(
            &node_covariances[6 * (node - nodes) + i], 
            (g_weights[point_idx] / node_total_weight) * unweighted[i]
        );
    }
}

__global__ void aggregate_node_position_sh_opacity_kernel(
    OctreeNode* nodes,
    float* node_centers,
    float* node_covariances,
    float* node_features,
    float* node_opacities,
    float* node_weight,
    float* node_total_weight,
    int features_dim,
    int node_count,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;

    OctreeNode* node = &nodes[idx];
    if (node->depth != depth) return;

    for (int i = 0; i < 8; ++i) {
        OctreeNode* child = node->children[i];

        // Check if the child pointer is valid
        if (child != nullptr) {
            // Calculate child weight based on scaling and opacity
            float child_cov[6] = {
                node_covariances[6 * (child - nodes) + 0], 
                node_covariances[6 * (child - nodes) + 1], 
                node_covariances[6 * (child - nodes) + 2],
                node_covariances[6 * (child - nodes) + 3], 
                node_covariances[6 * (child - nodes) + 4], 
                node_covariances[6 * (child - nodes) + 5]
            };
            float child_weight = calculate_weight(child_cov, node_opacities[child - nodes]);

            atomicAdd(&node_weight[child - nodes], child_weight);

            // Aggregate child attributes to parent node
            for (int j = 0; j < 3; ++j) {
                atomicAdd(
                    &node_centers[3 * (node - nodes) + j], 
                    child_weight * node_centers[3 * (child - nodes) + j]
                );
            }
            for (int j = 0; j < features_dim; ++j) {
                atomicAdd(
                    &node_features[features_dim * (node - nodes) + j], 
                    child_weight * node_features[features_dim * (child - nodes) + j]
                );
            }
            atomicAdd(&node_opacities[(node - nodes)], child_weight * node_opacities[(child - nodes)]);
            atomicAdd(&node_total_weight[node - nodes], child_weight);
        }
    }
}

__global__ void aggregate_node_covariance_kernel(
    OctreeNode* nodes,
    float* node_centers,
    float* node_covariances,
    float* node_opacity,
    float* total_weights,
    int node_count,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;

    OctreeNode* node = &nodes[idx];
    if (node->depth != depth) return;

    float node_total_weight = total_weights[node - nodes];

    for (int i = 0; i < 8; ++i) {
        OctreeNode* child = node->children[i];

        // Check if the child pointer is valid
        if (child != nullptr) {
            // Calculate child weight based on scaling and opacity
            float child_cov[6] = {
                node_covariances[6 * (child - nodes) + 0], 
                node_covariances[6 * (child - nodes) + 1], 
                node_covariances[6 * (child - nodes) + 2],
                node_covariances[6 * (child - nodes) + 3], 
                node_covariances[6 * (child - nodes) + 4], 
                node_covariances[6 * (child - nodes) + 5]
            };
            float child_weight = calculate_weight(child_cov, node_opacity[child - nodes]);

            float position_difference[3] = {
                node_centers[3 * (child - nodes)] - node_centers[3 * (node - nodes)],
                node_centers[3 * (child - nodes) + 1] - node_centers[3 * (node - nodes) + 1],
                node_centers[3 * (child - nodes) + 2] - node_centers[3 * (node - nodes) + 2]
            };

            float unweighted[6] = {
                child_cov[0] + position_difference[0] * position_difference[0],
                child_cov[1] + position_difference[0] * position_difference[1],
                child_cov[2] + position_difference[0] * position_difference[2],
                child_cov[3] + position_difference[1] * position_difference[1],
                child_cov[4] + position_difference[1] * position_difference[2],
                child_cov[5] + position_difference[2] * position_difference[2]
            };

            for (int i = 0; i < 6; ++i) {
                atomicAdd(
                    &node_covariances[6 * (node - nodes) + i], 
                    (child_weight / node_total_weight) * unweighted[i]
                );
            }
        }
    }
}

__global__ void build_octree_kernel(
    OctreeNode* nodes, 
    int* node_count,
    const float* points, 
    int num_points, 
    float* aabb_min, 
    float* aabb_max, 
    int max_depth,
    OctreeNode** point_to_node_pointer
) {
    // Global point index for each thread in the CUDA kernel
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    // Initial bounds and the root node are set for the point being processed by this thread.
    float current_min[3] = {aabb_min[0], aabb_min[1], aabb_min[2]};
    float current_max[3] = {aabb_max[0], aabb_max[1], aabb_max[2]};
    float point[3] = {points[point_idx * 3], points[point_idx * 3 + 1], points[point_idx * 3 + 2]};

    OctreeNode* current_node = &nodes[0];

    for (int depth = 0; depth < max_depth; ++depth) {
        // Calculate midpoint and determine octant
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

        // Atomic compare and swap to create a new child node if necessary
        // Create child node if it doesn't exist
        OctreeNode* expected_null = nullptr;
        OctreeNode* temp_non_null = reinterpret_cast<OctreeNode*>(0x1);
        if (atomicCAS(reinterpret_cast<unsigned long long int*>(&current_node->children[octant]),
              reinterpret_cast<unsigned long long int>(expected_null),
              reinterpret_cast<unsigned long long int>(temp_non_null)) == reinterpret_cast<unsigned long long int>(expected_null)) {

            int new_node_idx = atomicAdd(node_count, 1);
            
            nodes[new_node_idx] = OctreeNode(current_min, current_max, depth+1, current_node);
            current_node->children[octant] = &nodes[new_node_idx];

        } else {
            while (atomicCAS((unsigned long long int*)&current_node->children[octant],
                             (unsigned long long int)temp_non_null,
                             (unsigned long long int)temp_non_null) == (unsigned long long int)temp_non_null) {
                // Wait until the child pointer is set by another thread
            }
        }

        // Move to child node
        current_node = current_node->children[octant];

        // // Synchronize threads here so that nodes are created in the correct order from
        // // low depth to high depth
        // cudaDeviceSynchronize();
    }

    // If the current node is at max_depth, add it to max_depth_nodes array
    if (current_node->depth == max_depth) {
        point_to_node_pointer[point_idx] = current_node;
    }
}

__device__ __host__ OctreeNode::OctreeNode() {
    bbox = BoundingBox();
    depth = 0;
    parent = nullptr;

    for (int i = 0; i < 8; ++i) {
        children[i] = nullptr;
    }
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
}

Octree::Octree(
    const torch::Tensor points, 
    const torch::Tensor aabb_min, 
    const torch::Tensor aabb_max, 
    int max_depth
) : max_depth(max_depth) {
    
    CHECK_INPUT(points);
    CHECK_INPUT(aabb_min);
    CHECK_INPUT(aabb_max);
    
    cudaError_t err = cudaMalloc((void**)&device_nodes, sizeof(OctreeNode) * max_depth * points.size(0));
    CUDA_CHECK(err);

    // (max_depth + 1) because we consider root (0) until max_depth (max_depth) inclusive
    err = cudaMalloc((void**)&point_to_node_pointer, sizeof(OctreeNode*) * points.size(0));
    CUDA_CHECK(err);

    err = cudaMalloc((void**)&device_node_count, sizeof(int));
    CUDA_CHECK(err);

    err = cudaMalloc((void**)&device_node_per_depth_count, sizeof(int) * (max_depth + 1));
    CUDA_CHECK(err);

    // Initialize node_per_depth_count array's values to 0 for all depth levels
    err = cudaMemset(device_node_per_depth_count, 0, sizeof(int) * (max_depth + 1));
    CUDA_CHECK(err);

    // Set node count to 1 (root node)
    int initial_node_count = 1;
    err = cudaMemcpy(device_node_count, &initial_node_count, sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    cudaDeviceSynchronize();

    build(points, aabb_min, aabb_max);

    cudaDeviceSynchronize();
}

Octree::~Octree() {
    if (device_nodes) cudaFree(device_nodes);
    if (device_node_count) cudaFree(device_node_count);
    if (point_to_node_pointer) cudaFree(point_to_node_pointer);
    if (device_node_per_depth_count) cudaFree(device_node_per_depth_count);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Octree::build(
    const torch::Tensor points, 
    const torch::Tensor aabb_min, 
    const torch::Tensor aabb_max
) {
    num_points = points.size(0);

    // Convert torch::Tensor to float array
    float aabb_min_f3[3] = {aabb_min[0].item<float>(), aabb_min[1].item<float>(), aabb_min[2].item<float>()};
    float aabb_max_f3[3] = {aabb_max[0].item<float>(), aabb_max[1].item<float>(), aabb_max[2].item<float>()};

    // Initialize root node on host side
    OctreeNode root_node(aabb_min_f3, aabb_max_f3, 0, nullptr);
    cudaMemcpy(&device_nodes[0], &root_node, sizeof(OctreeNode), cudaMemcpyHostToDevice);

    // Launch CUDA kernel to build octree
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;
    build_octree_kernel<<<blocks, threads>>>(
        device_nodes,
        device_node_count,
        points.contiguous().data_ptr<float>(),
        num_points,
        aabb_min.contiguous().data_ptr<float>(),
        aabb_max.contiguous().data_ptr<float>(),
        max_depth,
        point_to_node_pointer);

    cudaDeviceSynchronize();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Octree::set_node_attributes(
    const torch::Tensor g_positions,
    const torch::Tensor g_covariances,
    const torch::Tensor g_features,
    const torch::Tensor g_opacities,
    const torch::Tensor g_weights,
    const torch::Tensor filtered_node_indices
) {
    CHECK_INPUT(g_positions);
    CHECK_INPUT(g_covariances);
    CHECK_INPUT(g_features);
    CHECK_INPUT(g_opacities);
    CHECK_INPUT(g_weights);
    CHECK_INPUT(filtered_node_indices);

    int num_points = g_positions.size(0);
    int num_filtered_points = filtered_node_indices.size(0);
    int num_nodes = get_node_count();
    int features_dim = g_features.size(1);
    int threads = 256;

    // Get minimum depth among filtered_node_indices
    auto min_depth_tensor = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int* min_depth = min_depth_tensor.data_ptr<int>();
    min_depth_tensor.fill_(max_depth);

    int blocks = (num_filtered_points + threads - 1) / threads;
    get_min_node_depth_kernel<<<blocks, threads>>>(
        device_nodes,
        filtered_node_indices.contiguous().data_ptr<int>(),
        min_depth,
        num_filtered_points
    );

    // Set the leaf node attributes
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor bbox_centers = torch::zeros({num_nodes, 3}, options);
    torch::Tensor covariances = torch::zeros({num_nodes, 6}, options);
    torch::Tensor features = torch::zeros({num_nodes, features_dim}, options);
    torch::Tensor opacities = torch::zeros({num_nodes}, options);
    torch::Tensor weights = torch::zeros({num_nodes}, options);
    torch::Tensor total_weights = torch::zeros({num_nodes}, options);

    blocks = (num_points + threads - 1) / threads;
    set_leaf_node_position_sh_opacity_kernel<<<blocks, threads>>>(
        device_nodes,
        g_positions.contiguous().data_ptr<float>(),
        g_features.contiguous().data_ptr<float>(),
        g_opacities.contiguous().data_ptr<float>(),
        g_weights.contiguous().data_ptr<float>(),
        bbox_centers.data_ptr<float>(),
        features.data_ptr<float>(),
        opacities.data_ptr<float>(),
        total_weights.data_ptr<float>(),
        point_to_node_pointer,
        features_dim,
        num_points,
        max_depth
    );

    blocks = (num_nodes + threads - 1) / threads;
    normalize_node_position_sh_opacity_at_depth_kernel<<<blocks, threads>>>(
        device_nodes,
        bbox_centers.data_ptr<float>(),
        features.data_ptr<float>(),
        opacities.data_ptr<float>(),
        total_weights.data_ptr<float>(),
        features_dim,
        num_nodes,
        max_depth
    );

    set_leaf_node_covariance_kernel<<<blocks, threads>>>(
        device_nodes,
        g_positions.contiguous().data_ptr<float>(),
        g_covariances.contiguous().data_ptr<float>(),
        g_weights.contiguous().data_ptr<float>(),
        bbox_centers.data_ptr<float>(),
        covariances.data_ptr<float>(),
        total_weights.data_ptr<float>(),
        point_to_node_pointer,
        num_points,
        max_depth
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    int host_min_depth;
    CUDA_CHECK(cudaMemcpy(&host_min_depth, min_depth, sizeof(int), cudaMemcpyDeviceToHost));

    // Aggregate node attributes from max_depth to 0
    for (int depth = max_depth - 1; depth >= host_min_depth; --depth) {
        aggregate_node_position_sh_opacity_kernel<<<blocks, threads>>>(
            device_nodes,
            bbox_centers.data_ptr<float>(),
            covariances.data_ptr<float>(),
            features.data_ptr<float>(),
            opacities.data_ptr<float>(),
            weights.data_ptr<float>(),
            total_weights.data_ptr<float>(),
            features_dim,
            num_nodes,
            depth
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Normalize node attributes at depth
        normalize_node_position_sh_opacity_at_depth_kernel<<<blocks, threads>>>(
            device_nodes,
            bbox_centers.data_ptr<float>(),
            features.data_ptr<float>(),
            opacities.data_ptr<float>(),
            total_weights.data_ptr<float>(),
            features_dim,
            num_nodes,
            depth
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        aggregate_node_covariance_kernel<<<blocks, threads>>>(
            device_nodes,
            bbox_centers.data_ptr<float>(),
            covariances.data_ptr<float>(),
            opacities.data_ptr<float>(),
            total_weights.data_ptr<float>(),
            num_nodes,
            depth
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return std::make_tuple(bbox_centers, covariances, features, opacities, weights, total_weights);
}

torch::Tensor Octree::get_nodes_at_depth(const int depth) const {
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor node_indices = torch::zeros({num_points}, options);

    get_nodes_at_depth_kernel<<<blocks, threads>>>(
        device_nodes,
        point_to_node_pointer,
        node_indices.contiguous().data_ptr<int>(),
        num_points,
        depth
    );

    // Synchronize to make sure the kernel has finished execution
    CUDA_CHECK(cudaDeviceSynchronize());

    return node_indices;
}

torch::Tensor Octree::get_nodes_w_max_granularity(
    const torch::Tensor projection_matrix, 
    const torch::Tensor view_matrix, 
    const int H, const int W,
    const float granularity_threshold
) const {
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor node_indices = torch::zeros({num_points}, options);

    filter_nodes_for_granularity_kernel<<<blocks, threads>>>(
        device_nodes,
        point_to_node_pointer,
        projection_matrix.contiguous().data_ptr<float>(),
        view_matrix.contiguous().data_ptr<float>(),
        H, W,
        node_indices.data_ptr<int>(),
        num_points,
        granularity_threshold
    );

    // Synchronize to make sure the kernel has finished execution
    cudaDeviceSynchronize();

    return node_indices;
}

torch::Tensor Octree::select_tree_cut(
    const torch::Tensor projection_matrix, 
    const torch::Tensor view_matrix, 
    const int H, const int W,
    const float granularity_threshold,
    const int chosen_depth
) const {

    // 1. Traverse tree from root to leaf nodes
    // 2. For each node, calculate the bounding box in 2D space
    // 3. If maximum dimension of the bounding box is less than granularity_threshold, add the node to the list
    // 4. If a node is added to the list, do not add its children (and their children) to the list
    
    if (chosen_depth != -1) {
        torch::Tensor node_indices = get_nodes_at_depth(chosen_depth);
        return node_indices;
    } else {
        torch::Tensor node_indices = get_nodes_w_max_granularity(
            projection_matrix, view_matrix, H, W, granularity_threshold
        );
        return node_indices;
        // int num_nodes = get_node_count();

        // // Allocate memory on device
        // int* device_filtered_node_count;
        // int* device_filtered_node_flags;
        // CUDA_CHECK(cudaMalloc((void**)&device_filtered_node_count, sizeof(int)));
        // CUDA_CHECK(cudaMalloc((void**)&device_filtered_node_flags, sizeof(int) * num_nodes));

        // // Set initial node count to 0 (root node)
        // int initial_node_count = 0;
        // CUDA_CHECK(cudaMemcpy(device_filtered_node_count, &initial_node_count, sizeof(int), cudaMemcpyHostToDevice));

        // // Launch kernels
        // int threads = 256;
        // int blocks = (num_nodes + threads - 1) / threads;

        // CUDA_CHECK(cudaDeviceSynchronize());

        // // Prune descendants in a depthwise manner
        // for (int depth = max_depth; depth >= 0; --depth) {
        //     filter_and_prune_nodes_for_granularity_kernel<<<blocks, threads>>>(
        //         device_nodes,
        //         projection_matrix.contiguous().data_ptr<float>(),
        //         view_matrix.contiguous().data_ptr<float>(),
        //         H, W,
        //         device_filtered_node_flags,
        //         device_filtered_node_count,
        //         num_nodes,
        //         granularity_threshold,
        //         depth
        //     );
        //     CUDA_CHECK(cudaDeviceSynchronize());
        // }

        // // Copy results from device to host
        // int filtered_node_count;
        // CUDA_CHECK(
        //     cudaMemcpy(
        //         &filtered_node_count, 
        //         device_filtered_node_count, 
        //         sizeof(int), cudaMemcpyDeviceToHost
        //     )
        // );

        // CUDA_CHECK(cudaDeviceSynchronize());

        // // Allocate memory for filtered node indices. Build a torch::Tensor for it on CUDA
        // auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        // torch::Tensor filtered_node_indices = torch::empty({filtered_node_count}, options);
        // // int* device_filtered_node_indices;
        // // CUDA_CHECK(cudaMalloc((void**)&device_filtered_node_indices, sizeof(int) * filtered_node_count));

        // // Get indices of filtered nodes
        // get_node_indices_from_flags_kernel<<<blocks, threads>>>(
        //     device_filtered_node_flags,
        //     filtered_node_indices.data_ptr<int>(),
        //     device_filtered_node_count, // Subtract from device_filtered node count to have a dense list of indices
        //     num_nodes
        // );
        // CUDA_CHECK(cudaDeviceSynchronize());
        // CUDA_CHECK(cudaFree(device_filtered_node_count))
        // CUDA_CHECK(cudaFree(device_filtered_node_flags));

        // return filtered_node_indices;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Octree::set_node_attributes_backward(
    const torch::Tensor dL_dnew_position,
    const torch::Tensor dL_dnew_covariances,
    const torch::Tensor dL_dnew_features,
    const torch::Tensor dL_dnew_opacities,
    const torch::Tensor node_indices,
    const torch::Tensor point_weights,
    const torch::Tensor weights,
    const torch::Tensor total_weights
) const {
    CHECK_INPUT(dL_dnew_position);
    CHECK_INPUT(dL_dnew_covariances);
    CHECK_INPUT(dL_dnew_features);
    CHECK_INPUT(dL_dnew_opacities);
    CHECK_INPUT(node_indices);
    CHECK_INPUT(point_weights);
    CHECK_INPUT(weights);
    CHECK_INPUT(total_weights);

    int num_new_nodes = dL_dnew_position.size(0);
    int features_dim = dL_dnew_features.size(1);
    int num_nodes = get_node_count();
    int num_points = point_weights.size(0);

    int threads = 256;
    int blocks = (num_new_nodes + threads - 1) / threads;

    auto min_depth_of_new_nodes_tensor = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int* min_depth_of_new_nodes = min_depth_of_new_nodes_tensor.data_ptr<int>();
    min_depth_of_new_nodes_tensor.fill_(max_depth);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor dL_dposition = torch::zeros({num_nodes, 3}, options);
    torch::Tensor dL_dcovariances = torch::zeros({num_nodes, 6}, options);
    torch::Tensor dL_dfeatures = torch::zeros({num_nodes, features_dim}, options);
    torch::Tensor dL_dopacities = torch::zeros({num_nodes, 1}, options);

    propagate_output_node_gradients_kernel<<<blocks, threads>>>(
        device_nodes,
        dL_dnew_position.contiguous().data_ptr<float>(),
        dL_dnew_covariances.contiguous().data_ptr<float>(),
        dL_dnew_features.contiguous().data_ptr<float>(),
        dL_dnew_opacities.contiguous().data_ptr<float>(),
        dL_dposition.data_ptr<float>(),
        dL_dcovariances.data_ptr<float>(),
        dL_dfeatures.data_ptr<float>(),
        dL_dopacities.data_ptr<float>(),
        node_indices.contiguous().data_ptr<int>(),
        min_depth_of_new_nodes,
        features_dim,
        num_new_nodes
    );

    int host_min_depth_of_new_nodes;
    CUDA_CHECK(
        cudaMemcpy(
            &host_min_depth_of_new_nodes, 
            min_depth_of_new_nodes, 
            sizeof(int), cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    blocks = (num_nodes + threads - 1) / threads;

    for (int depth = host_min_depth_of_new_nodes; depth < max_depth; ++depth) {        
        // Propagate gradients from parent to children
        propagate_gradients_from_parent_to_child<<<blocks, threads>>>(
            device_nodes,
            dL_dposition.data_ptr<float>(),
            dL_dcovariances.data_ptr<float>(),
            dL_dfeatures.data_ptr<float>(),
            dL_dopacities.data_ptr<float>(),
            weights.contiguous().data_ptr<float>(),
            total_weights.contiguous().data_ptr<float>(),
            features_dim, num_nodes, depth
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Propagating gradients to points
    options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out_dL_dposition = torch::zeros({num_points, 3}, options);
    torch::Tensor out_dL_dcovariances = torch::zeros({num_points, 6}, options);
    torch::Tensor out_dL_dfeatures = torch::zeros({num_points, features_dim}, options);
    torch::Tensor out_dL_dopacities = torch::zeros({num_points, 1}, options);
    
    blocks = (num_points + threads - 1) / threads;
    propagate_gradients_from_leaf_node_to_points<<<blocks, threads>>>(
        device_nodes,
        dL_dposition.data_ptr<float>(),
        dL_dcovariances.data_ptr<float>(),
        dL_dfeatures.data_ptr<float>(),
        dL_dopacities.data_ptr<float>(),
        out_dL_dposition.data_ptr<float>(),
        out_dL_dcovariances.data_ptr<float>(),
        out_dL_dfeatures.data_ptr<float>(),
        out_dL_dopacities.data_ptr<float>(),
        point_weights.contiguous().data_ptr<float>(),
        total_weights.contiguous().data_ptr<float>(),
        point_to_node_pointer,
        features_dim, num_points
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(out_dL_dposition, out_dL_dcovariances, out_dL_dfeatures, out_dL_dopacities);
}

int Octree::get_node_count() const {
    int node_count;
    cudaMemcpy(&node_count, device_node_count, sizeof(int), cudaMemcpyDeviceToHost);
    return node_count;
}

std::vector<int> Octree::get_node_per_depth_count() const {
    std::vector<int> node_per_depth_vec(max_depth + 1);
    cudaMemcpy(
        node_per_depth_vec.data(), device_node_per_depth_count, 
        sizeof(int) * (max_depth + 1), cudaMemcpyDeviceToHost
    );

    return node_per_depth_vec;
}

OctreeNode Octree::get_node_by_id(int node_id) const {
    if (node_id >= get_node_count()) {
        throw std::out_of_range("Node ID out of range");
    }
    OctreeNode node;
    cudaMemcpy(&node, &device_nodes[node_id], sizeof(OctreeNode), cudaMemcpyDeviceToHost);
    return node;
}