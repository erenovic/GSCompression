import os

from torch.utils.cpp_extension import load

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = '1'


def setup():
    sources = [
        "submodules/build_octree/lib/build_octree.cu",
        "submodules/build_octree/ext.cpp",
    ]

    include_dirs = []

    octree_generation_module = load(
        name="build_octree",
        sources=sources,
        extra_include_paths=include_dirs,
        extra_cuda_cflags=["-DTORCH_USE_CUDA_DSA=1", "-lineinfo"],
        verbose=True,
    )
