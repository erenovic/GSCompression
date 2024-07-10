import os

from torch.utils.cpp_extension import load

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = '1'


def setup():
    sources = [
        "submodules/hierarchy_generation/lib/octree.cu",
        "submodules/hierarchy_generation/lib/octree.h",
        "submodules/hierarchy_generation/ext.cpp",
    ]

    include_dirs = []

    hierarchy_generation_module = load(
        name="hierarchy_generation",
        sources=sources,
        extra_include_paths=include_dirs,
        extra_cuda_cflags=["-DTORCH_USE_CUDA_DSA=1", "-lineinfo"],
        verbose=True,
    )
