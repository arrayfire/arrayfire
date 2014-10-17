## Requirements

### General
* gcc >= 4.7
* cmake >= 2.8.9
* svn >= 1.6
* freeimage

### CPU Backend
* atlas on Linux
* Accelerate Framework on OSX

### CUDA Backend
* CUDA toolkit >= 6.5

### OpenCL Backend
* CUDA Toolkit >= 6.5 or AMDAPPSDK >= 2.9 or Intel OpenCL SDK >= 4.4
* [clBLAS](http://github.com/clMathLibraries/clBLAS)
* [clFFT](http://github.com/clMathLibraries/clFFT)
* [Boost.Compute](http://github.com/kylelutz/compute)
* [Boost](http://boost.org) >= 1.48

## Building clBLAS and clFFT
The CMake program currently expects the clBLAS library to be located in the same directory as the ArrayFire repository. The clBLAS library needs to be built and installed in the build/package folder of the project(this is the default behavior). Here are the steps you will need to take to build the clBLAS project.

In the commands below, simply replace clBLAS with clFFT to build and install clFFT

```bash
git clone http://github.com/clMathLibraries/clBLAS.git
cd clBLAS
mkdir build
cd build
cmake ../src -DCMAKE_BUILD_TYPE:STRING=Release
make
make install
```

## Building ArrayFire

```bash
mkdir build && cd build
ccmake ..
make
make test
```

Note that only CPU backend is enabled by default. Enable the other backends as necessary after running the ccmake command.

## Common Issues
If your compiler cannot find the cblas_* symbols when linking, make sure the cblas library that CMake found is correct. You can set the correct cblas library in ccmake.

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
