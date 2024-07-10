import os

os.path.dirname(os.path.abspath(__file__))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hierarchy_generation",
    ext_modules=[CUDAExtension(name="hierarchy_generation", sources=["ext.cpp", "lib/octree.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
