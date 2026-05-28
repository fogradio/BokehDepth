import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

# Disable the CUDA version check (monkey-patch).
# Architectures compatible with CUDA 11.7.
os.environ['TORCH_CUDA_ARCH_LIST'] = os.environ.get('TORCH_CUDA_ARCH_LIST', '6.1 7.0 7.5 8.0 8.6')
import torch.utils.cpp_extension
original_check = torch.utils.cpp_extension._check_cuda_version
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]

        # Check PyTorch's CXX11 ABI setting.
        cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI
        print(f">>> PyTorch CXX11 ABI: {cxx11_abi}")

        extra_compile_args["nvcc"] = [
            "-O3",
            "--allow-unsupported-compiler",  # allow newer GCC versions
        ]
        # Use the same ABI setting as PyTorch.
        extra_compile_args["cxx"] = [
            "-O3",
            f"-D_GLIBCXX_USE_CXX11_ABI={1 if cxx11_abi else 0}"
        ]
    else:
        raise NotImplementedError("Cuda is not available")

    sources = list(set([os.path.join(extensions_dir, s) for s in sources]))
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "KNN",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="KNN",
    version="0.1",
    author="Luigi Piccinelli",
    ext_modules=get_extensions(),
    packages=find_packages(
        exclude=(
            "configs",
            "tests",
        )
    ),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
