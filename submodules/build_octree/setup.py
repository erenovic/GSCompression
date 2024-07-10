#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.path.dirname(os.path.abspath(__file__))

setup(
    name="build_octree",
    packages=["lib"],
    ext_modules=[
        CUDAExtension(
            name="build_octree",
            sources=["lib/octree.cu", "ext.cpp"],
            extra_compile_args={},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
