from conans import ConanFile, CMake, tools
import os


ARRAYFIRE_VERSION = "3.7.1"
BINARY_INSTALLER_NAME_SUFFIX = "-1"
BINARY_INSTALLER_NAME = f"ArrayFire-v{ARRAYFIRE_VERSION}{BINARY_INSTALLER_NAME_SUFFIX}_Linux_x86_64.sh"
CUDA_TOOLKIT_VERSION = "10.0"

class ArrayFireConan(ConanFile):
    name = "arrayfire"
    version = ARRAYFIRE_VERSION
    license = "BSD"
    author = "jacobkahn jacobkahn1@gmail.com"
    url = "https://github.com/arrayfire/arrayfire"
    requires = []
    description = "ArrayFire: a general purpose GPU library"
    topics = ("arrayfire", "gpu", "cuda", "opencl", "gpgpu",
              "hpc", "performance", "scientific-computing")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "cpu_backend": [True, False],
        "cuda_backend": [True, False],
        "opencl_backend": [True, False],
        "unified_backend": [True, False],
        "graphics": [True, False],
    }
    generators = "cmake"  # unused

    def configure(self):
        if self.settings.os == "Windows":
            raise ConanInvalidConfiguration(
                "Linux binary installer not compaible with Windows.")

    def requirements(self):
        if self.options.graphics:
            self.requires('glfw/3.3.2@bincrafters/stable')

    def _download_arrayfire(self):
        self.af_installer_local_path = BINARY_INSTALLER_NAME
        if not os.path.exists(self.af_installer_local_path):
            self.output.info(
                f"Downloading the ArrayFire {ARRAYFIRE_VERSION} binary installer...")
            tools.download(
                f"https://arrayfire.s3.amazonaws.com/{ARRAYFIRE_VERSION}/{BINARY_INSTALLER_NAME}", self.af_installer_local_path)
            self.output.success(
                f"ArrayFire {ARRAYFIRE_VERSION} binary installer successfully downloaded to {self.af_installer_local_path}")
        else:
            self.output.info(
                f"ArrayFire {ARRAYFIRE_VERSION} binary installer already exists - skipping download.")

    def _unpack_arrayfire(self):
        if not os.path.exists(self.af_unpack_path):
            os.mkdir(self.af_unpack_path)
        self.output.info(
            f"Unpacking ArrayFire {ARRAYFIRE_VERSION} binary installer...")
        cmd = f"bash {self.af_installer_local_path} --prefix={self.af_unpack_path} --skip-license"
        self.run(cmd)
        self.output.success(
            f"ArrayFire {ARRAYFIRE_VERSION} successfully unpacked.")

    def _process_arrayfire(self):
        # Install ArrayFire to requisite path
        self.af_unpack_path = os.path.join(self.source_folder, 'arrayfire')

        # Only proceed if missing
        if os.path.exists(os.path.join(self.af_unpack_path, 'include', 'arrayfire.h')):
            self.output.info(
                f"ArrayFire {ARRAYFIRE_VERSION} already unpacked - skipping.")
        else:
            self._download_arrayfire()
            self._unpack_arrayfire()

    def build(self):
        self._process_arrayfire()

    def package(self):
        # libs
        self.copy("*.so", dst="lib", keep_path=False, symlinks=True)
        self.copy("*.so.*", dst="lib", keep_path=False, symlinks=True)

        # headers
        self.copy("*.h", dst="include", src="arrayfire/include")
        self.copy("*.hpp", dst="include", src="arrayfire/include")

    def package_info(self):
        self.cpp_info.libs = []
        if self.options.unified_backend:
            self.cpp_info.libs.extend([
                f"libaf.so.{ARRAYFIRE_VERSION}",
            ])
        if self.options.graphics:
            self.cpp_info.libs.extend([
                "libforge.so.1.0.5",
            ])
        if self.options.cuda_backend:
            self.cpp_info.libs.extend([
                f"libafcuda.so.{ARRAYFIRE_VERSION}",
                "libnvrtc-builtins.so",
                f"libcudnn.so.{CUDA_TOOLKIT_VERSION}",
                f"libcusparse.so.{CUDA_TOOLKIT_VERSION}",
                f"libcublas.so.{CUDA_TOOLKIT_VERSION}",
                f"libcusolver.so.{CUDA_TOOLKIT_VERSION}",
                f"libnvrtc.so.{CUDA_TOOLKIT_VERSION}",
                f"libcufft.so.{CUDA_TOOLKIT_VERSION}",
            ])
        if self.options.cpu_backend:
            self.cpp_info.libs.extend([
                f"libafcpu.so.{ARRAYFIRE_VERSION}",
                "libmkl_avx2.so",
                "libmkl_mc.so",
                "libmkl_intel_lp64.so",
                "libmkl_core.so",
                "libmkl_avx.so",
                "libmkl_def.so",
                "libiomp5.so",
                "libmkl_avx512.so",
                "libmkl_intel_thread.so",
                "libmkl_mc3.so",

            ])
        if self.options.opencl_backend:
            self.cpp_info.libs.extend([
                f"libafopencl.so.{ARRAYFIRE_VERSION}",
                "libOpenCL.so.1",
            ])
