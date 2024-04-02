import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='cartpole2l',
    ext_modules=[
        CUDAExtension('cartpole2l', 
            include_dirs=[osp.join(ROOT, 'include')],
            sources=['src/dynamics.cpp', 'src/dynamics_cpu.cpp', 'src/dynamics_gpu.cu', 
                     'src/generated_dynamics.c', 'src/generated_derivatives.c'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
