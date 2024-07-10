#include <torch/extension.h>
#include "lib/octree.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        // .def(py::init<const float[3], const float[3]>())
        .def_readwrite("min", &BoundingBox::min)
        .def_readwrite("max", &BoundingBox::max)
        .def_readwrite("size", &BoundingBox::size)
        .def_readwrite("center", &BoundingBox::center)
        .def("inside", &BoundingBox::inside);

    py::class_<OctreeNode>(m, "OctreeNode")
        .def(py::init<>())
        .def_readwrite("bbox", &OctreeNode::bbox)
        .def_readwrite("children", &OctreeNode::children)
        .def_readwrite("depth", &OctreeNode::depth);
        // .def("get_node_position", &OctreeNode::get_node_position)
        // .def("get_children_indices", &OctreeNode::get_children_indices)
        // .def("get_children", &OctreeNode::get_children)
        // .def("get_parent", &OctreeNode::get_parent)
        // .def("get_opacity", &OctreeNode::get_opacity)
        // .def("get_scaling", &OctreeNode::get_scaling)
        // .def("get_rotation", &OctreeNode::get_rotation)
        // .def("get_features", &OctreeNode::get_features)
        // .def("get_weight", &OctreeNode::get_weight)
        // .def("isLeaf", &OctreeNode::isLeaf);

    py::class_<Octree>(m, "Octree")
        .def_readwrite("max_depth", &Octree::max_depth)
        .def_readwrite("device_nodes", &Octree::device_nodes)
        .def(py::init<const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int>())
        .def("get_node_by_id", &Octree::get_node_by_id)
        .def("get_node_count", &Octree::get_node_count)
        .def("get_node_per_depth_count", &Octree::get_node_per_depth_count)
        .def("build", &Octree::build)
        .def("set_node_attributes", &Octree::set_node_attributes)
        .def("set_node_attributes_backward", &Octree::set_node_attributes_backward)
        .def("select_tree_cut", &Octree::select_tree_cut);
        // .def("get_nodes_at_depth", &Octree::get_nodes_at_depth);
}
