from conans import ConanFile, CMake, tools
import os


class ArrayFireConan(ConanFile):
    name = "arrayfire"
    version = "3.7.1"
    license = "BSD"
    author = "jacobkahn jacobkahn1@gmail.com"
    url = "https://github.com/arrayfire/arrayfire"
    requires = []
    description = "ArrayFire: a general purpose GPU library"
    topics = ("arrayfire", "gpu", "cuda", "opencl", "gpgpu",
              "hpc", "performance", "scientific-computing")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "backend": ["CPU", "CUDA", "OPENCL"]}
    default_options = {"shared": True}
    generators = "cmake"  # unused

    def configure(self):
        if self.settings.os == "Windows":
            raise ConanInvalidConfiguration(
                "Linux binary installer not compaible with Windows.")

    def _download_arrayfire(self):
        # Download ArrayFire 3.7.1
        self.af_installer_local_path = 'ArrayFire-v3.7.1-1_Linux_x86_64.sh'
        if not os.path.exists(self.af_installer_local_path):
            self.output.info(
                "Downloading the ArrayFire 3.7.1 binary installer...")
            tools.download(
                'https://arrayfire.s3.amazonaws.com/3.7.1/ArrayFire-v3.7.1-1_Linux_x86_64.sh', self.af_installer_local_path)
            self.output.success(
                f"ArrayFire 3.7.1 binary installer successfully downloaded to {self.af_installer_local_path}")
        else:
            self.output.info(
                "ArrayFire 3.7.1 binary installer already exists - skipping download.")

    def _unpack_arrayfire(self):
        if not os.path.exists(self.af_unpack_path):
            os.mkdir(self.af_unpack_path)
        self.output.info("Unpacking ArrayFire 3.7.1 binary installer...")
        cmd = f"bash {self.af_installer_local_path} --prefix={self.af_unpack_path} --skip-license"
        self.run(cmd)
        self.output.success("ArrayFire 3.7.1 successfully unpacked.")

    def _process_arrayfire(self):
        # Install ArrayFire to requisite path
        self.af_unpack_path = os.path.join(self.source_folder, 'arrayfire')

        # Only proceed if missing
        if os.path.exists(os.path.join(self.af_unpack_path, 'include', 'arrayfire.h')):
            self.output.info("ArrayFire 3.7.1 already unpacked - skipping.")
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
        self.cpp_info.libs = [
            "libaf.so.3.7.1",
            "libforge.so.1.0.5",
        ]
        if self.options.backend == 'CUDA':
            self.cpp_info.libs.extend([
                "libnvrtc-builtins.so",
                "libcudnn.so.10.0",
                "libcusparse.so.10.0",
                "libcublas.so.10.0",
                "libafcuda.so.3.7.1",
                "libcusolver.so.10.0",
                "libnvrtc.so.10.0",
                "libcufft.so.10.0",
            ])
        elif self.options.backend == 'CPU':
            self.cpp_info.libs.extend([
                "libmkl_avx2.so",
                "libmkl_mc.so",
                "libmkl_intel_lp64.so",
                "libmkl_core.so",
                "libmkl_avx.so",
                "libmkl_def.so",
                "libiomp5.so",
                "libafopencl.so.3.7.1",
                "libmkl_avx512.so",
                "libmkl_intel_thread.so",
                "libmkl_mc3.so",
                "libafcpu.so.3.7.1",
            ])
        elif self.options.backend == 'OPENCL':
            self.cpp_info.libs.extend([
                "libOpenCL.so.1",
            ])
