import glob
import os
import platform
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Any, List, Union

import setuptools

PLATFORM = platform.system()
PROJECT_NAME = "face-to-face"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODULE_NAME = "f2f"
SOURCE_DIR = os.path.join(PROJECT_ROOT, MODULE_NAME)


def _load_py_module(filename: str, pkg: str = MODULE_NAME) -> ModuleType:
    spec = spec_from_file_location(
        os.path.join(pkg, filename), os.path.join(SOURCE_DIR, filename)
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


VERSION = _load_py_module("__version__.py").__version__


if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"


def is_success(status: int) -> bool:
    return status == 0


def _find_windows_compiler() -> Union[str, None]:
    compiler_paths = [
        "C:/Program Files (x86)/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64",  # noqa
        "C:/Program Files (x86)/Microsoft Visual Studio */vc/bin",
    ]
    for compiler_path in compiler_paths:
        found_paths = glob.glob(compiler_path)
        if found_paths:
            return sorted(found_paths)[-1]
    return None


def get_extensions() -> List[Any]:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

    print(f"Found torch version: {torch.__version__}")

    extensions_dir = os.path.join(SOURCE_DIR, "csrc")
    sources = glob.glob(
        os.path.join(extensions_dir, "**", "*.cpp"), recursive=True
    )
    sources_cuda = glob.glob(
        os.path.join(extensions_dir, "**", "*.cu"), recursive=True
    )
    sources += sources_cuda
    if not sources:
        return []

    extra_compile_args = {"cxx": ["-Wall", "-std=c++14"]}

    extension = CppExtension

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    cuda_available = torch.cuda.is_available() and CUDA_HOME is not None
    if not (force_cuda or cuda_available):
        print("CUDA not found, skipping CUDA extensions")
        return []

    print(f"CUDA is available, found {CUDA_HOME}")
    extension = CUDAExtension
    if PLATFORM == "Windows" and not is_success(
        os.system("where cl.exe >nul 2>nul")
    ):
        found_compiler = _find_windows_compiler()
        if found_compiler is None:
            raise RuntimeError(
                "Cannot find compiler like MSVC/GCC."
                "Make sure it's installed and in the PATH"
            )
        os.environ["PATH"] = found_compiler + ";" + os.environ["PATH"]

    nvcc_args = [
        "-std=c++14",
        "-Xcompiler",
        "-Wall",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-gencode",
        "arch=compute_75,code=sm_75",
    ]
    if torch.version.cuda is not None and torch.version.cuda.startswith("11"):
        nvcc_args.extend(["-gencode", "arch=compute_80,code=sm_80"])
    extra_compile_args["nvcc"] = nvcc_args

    extensions = [
        extension(
            name=f"{MODULE_NAME}._C",
            sources=[
                os.path.relpath(source, PROJECT_ROOT) for source in sources
            ],
            include_dirs=[extensions_dir],
            extra_compile_args=extra_compile_args,
        )
    ]
    return extensions


install_requires = ["ninja", "onnxruntime-gpu", "Pillow", "tqdm"]

try:
    from torch.utils.cpp_extension import BuildExtension

    ext_modules = get_extensions()
    cmd_class = {"build_ext": BuildExtension}
except ImportError:
    ext_modules = []
    cmd_class = {}

cuda_sources = [
    os.path.relpath(source, SOURCE_DIR)
    for source in glob.glob(
        os.path.join(SOURCE_DIR, "csrc", "**", "*"), recursive=True
    )
]
setuptools.setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=f"{PROJECT_NAME} team",
    author_email="kynk94@naver.com",
    description=(
        f"{PROJECT_NAME} is a library for easy-to-use human face related "
        "trained models."
    ),
    url=f"https://github.com/kynk94/{PROJECT_NAME}",
    license="MIT",
    packages=setuptools.find_packages(include=[f"{MODULE_NAME}*"]),
    package_data={MODULE_NAME: cuda_sources},
    install_requires=install_requires,
    extras_require={},
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules=ext_modules,
    cmdclass=cmd_class,
)
