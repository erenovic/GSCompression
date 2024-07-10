/*
 * Eren Cetin
 */

#include <torch/extension.h>
#include "lib/octree.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Octree>(m, "Octree")
        .def_readwrite("max_depth", &Octree::max_depth)
        .def(py::init<int>())
        .def("get_point_node_assignment", &Octree::get_point_node_assignment, "Get Point Node Assignment CUDA kernel")
        .def("generate_octree", &Octree::generate_octree, "Build Octree CUDA kernel");
    
    m.def("aggregate_gaussians_fwd", &aggregate_gaussians_fwd, "Aggregate Gaussians Forward CUDA kernel");
    m.def("aggregate_gaussians_bwd", &aggregate_gaussians_bwd, "Aggregate Gaussians Backward CUDA kernel");
}