## Description

ArrayFire is a fast, hardware-neutral software library for GPU computing with
an easy-to-use API. Its array-based function set makes GPU programming simple.
A few lines of code in ArrayFire can replace dozens of lines of raw GPU code,
saving you valuable time and lowering development costs.

## Prerequisites

### General

* gcc >= 4.7 (gcc, g++)
* cmake >= 2.8.9 (cmake, cmake-curses-gui)
* git >= 1.8
* svn >= 1.6 (subversion)
* freeimage (libfreeimage-dev)
* Linux / OSX (Windows coming soon)

### CPU Backend
* atlas on Linux (libatlas3gf-base, libatlas-dev)
* Accelerate Framework on OSX
* fftw3

### CUDA Backend

* CUDA toolkit >= 6.5

### OpenCL Backend
* An OpenCL SDK
  * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 6.5
  * [AMDAPPSDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/) >= 2.9
  * [Intel OpenCL SDK](https://software.intel.com/en-us/intel-opencl) >= 4.4
* ArrayFire fork of [clBLAS](http://github.com/arrayfire/clBLAS)
* ArrayFire fork of [clFFT](http://github.com/arrayfire/clFFT)
* [Boost.Compute](http://github.com/kylelutz/compute)
* [Boost](http://boost.org) >= 1.48

#### Building clBLAS and clFFT

Our CMake scripts expect the ArrayFire, clBLAS, and clFFT git clones to
reside within the same parent folder. Prior to building ArrayFire's OpenCL backend
both `clBLAS` and `clFFT` need to be built and have their `make install` steps
executed to create packages against which ArrayFire may link.
Complete the following steps for both `clBLAS` and `clFFT` (replacing the
epository names when needed):

```bash
git clone http://github.com/arrayfire/clBLAS.git
cd clBLAS
mkdir build
cd build
cmake ../src -DCMAKE_BUILD_TYPE:STRING=Release
make
make install
```

## Getting ArrayFire

``` bash
git clone --recursive git@github.com:arrayfire/arrayfire.git
```
Do not forget to include `--recursive` as this clones the dependent libraries.

## Building ArrayFire

```bash
mkdir build && cd build
ccmake ..
make
make test
```

Note that only CPU backend is enabled by default. Enable the other backends as
necessary after running the ccmake command.

## Common Issues
If your compiler cannot find the cblas_* symbols when linking, make sure the
cblas library that CMake found is correct. You can set the correct cblas
library in ccmake.

### CentOS 6.*
- Install devtoolset-2 using the instructions from [here](http://people.centos.org/tru/devtools-2/readme).

```bash
scl enable detoolset-2 bash
ccmake ..
make
make test
```

- Required version of Boost (>=1.48) may not be available using the package
  manager. It needs to be downloaded and installed. The following commands are
  for version 1.55. For a different version, simply change the version number.
  You may also choose a --perfix path of your choosing, but may need to
  manually edit the cmake path if it is not in one of the standard locations.

```bash
wget http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz
tar -xvzf boost_1_55_0.tar.gz
cd boost_1_55_0/
./bootstrap.sh --prefix=/usr/local
./b2 install --with-all
```
