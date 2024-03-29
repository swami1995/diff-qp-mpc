#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import os
import setuptools
import sys

try:
    import torch
    from torch.utils import cpp_extension as torch_cpp_ext

    # This is a hack to get CUDAExtension to compile with
    #   -ltorch_cuda
    # instead of split
    #   -ltorch_cuda_cu -ltorch_cuda_cpp
    torch_cpp_ext.BUILD_SPLIT_CUDA = False

    # hack to be able to compile with gcc-8.4.0
    torch_cpp_ext.CUDA_GCC_VERSIONS["10.2"] = (
        torch_cpp_ext.MINIMUM_GCC_VERSION,
        (8, 4, 99),
    )
except ModuleNotFoundError:
    print("Installation requires torch.")
    sys.exit(1)


def parse_requirements_file(path):
    with open(path) as f:
        reqs = []
        for line in f:
            line = line.strip()
            reqs.append(line.split("==")[0])
    return reqs

reqs_main = parse_requirements_file("requirements/main.txt")
root_dir = Path(__file__).parent

with open("README.md", "r") as fh:
    long_description = fh.read()

# Add C++ and CUDA extensions
compile_cuda_flag = os.environ.get("FORCE_CUDA")
compile_cuda_support = (
    torch.cuda.is_available()
    if (compile_cuda_flag is None)
    else (compile_cuda_flag not in {"", "0", "False"})
)
cuda_detection_info = (
    "detected" if compile_cuda_flag is None else "forced by FORCE_CUDA env var"
)
print(f"CUDA support: {compile_cuda_support} ({cuda_detection_info})")

if compile_cuda_support:
    ext_modules = [
        # reference: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
        # torch_cpp_ext.CUDAExtension(
        #     name="theseus.extlib.mat_mult",
        #     sources=[str(root_dir / "theseus" / "extlib" / "mat_mult.cu")],
        # ),
        torch_cpp_ext.CUDAExtension(
            name="qpth.extlib.cusolver_lu_solver",
            sources=[
                str(root_dir / "qpth" / "extlib" / "cusolver_lu_solver.cpp"),
                str(root_dir / "qpth" / "extlib" / "cusolver_sp_defs.cpp"),
            ],
            include_dirs=[str(root_dir)],
            libraries=["cusolver"],
        ),
    ]
else:
    print("No CUDA support found. CUDA extensions won't be installed.")
    ext_modules = []

setuptools.setup(
    name="qpth",
    version='0.0.1',
    author="Swaminathan Gurumurthy",
    description="A library for differentiable nonlinear MPC.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swami1995/qpth",
    keywords="differentiable optimization, MPC, Sparse solvers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=reqs_main,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    ext_modules=ext_modules,
)
