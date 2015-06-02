Using ArrayFire on Linux {#using_on_linux}
=====


Among the many possible build systems on Linux we suggest using ArrayFire with
either CMake or Makefiles with CMake being the preferred build system.

## Pre-requisites

Before you get started, make sure you have the necessary pre-requisites.

- If you are using CUDA, please make sure you have [CUDA 7](https://developer.nvidia.com/cuda-downloads) installed on your system.
     - [Contact us](support@arrayfire.com) for custom builds (eg. different toolkits)

- If you are using OpenCL, please make sure you have one of the following SDKs.
     - [AMD OpenCL SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
     - [Intel OpenCL SDK](https://software.intel.com/en-us/articles/download-the-latest-intel-amt-software-development-kit-sdk)
     - [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

You will also need the following dependencies to use ArrayFire.

#### Fedora, Centos and Redhat

Install EPEL repo (not required for Fedora)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
yum install epel-release
yum update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the common dependencies

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
yum install gcc gcc-c++ cmake make
yum install freeimage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install glfw (not required for no-gl installers)

Fedora:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
yum install glfw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Centos and Redhat, please follow [these instructions](https://github.com/arrayfire/arrayfire/wiki/GLFW-for-ArrayFire)

#### Debian and Ubuntu

Install common dependencies

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
apt-get install build-essential cmake cmake-curses-gui
apt-get libfreeimage3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install glfw (not required for no-gl installers)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
apt-get install libglfw3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Debian 7 and Ubuntu 14.04, please follow [these instructions](https://github.com/arrayfire/arrayfire/wiki/GLFW-for-ArrayFire)

**Special instructions for Tegra-K1**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sudo apt-get install libatlas3gf-base libatlas-dev libfftw3-dev liblapacke-dev
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## CMake

This is the suggested method of using ArrayFire on Linux.
ArrayFire ships with support for CMake by default, including a series of
`Find` scripts installed  in the `/usr/local/share/ArrayFire` (or similar)
directory.
These scripts will automatically find the CUDA, OpenCL, and CPU versions
of ArrayFire and automatically choose the most powerful installed backend
(typically CUDA).

To use ArrayFire, simply insert the `FIND_PACKAGE` command inside of your
`CMakeLists.txt` file as follows:

    FIND_PACKAGE(ArrayFire)
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
    ...

    ADD_EXECUTABLE(some_executable ...)
    TARGET_LINK_LIBRARIES(some_executable ${ArrayFire_LIBRARIES} )

The find script will automatically define several variables including:

    ArrayFire_INCLUDE_DIRS    - Location of ArrayFire's include directory.
    ArrayFire_LIBRARIES       - Location of ArrayFire's libraries. This will default
                                to a GPU backend if one
    ArrayFire_FOUND           - True if ArrayFire has been located

If you wish to use a specific backend, the find script also defines these variables:

    ArrayFire_CPU_FOUND        - True of the ArrayFire CPU library has been found.
    ArrayFire_CPU_LIBRARIES    - Location of ArrayFire's CPU library, if found
    ArrayFire_CUDA_FOUND       - True of the ArrayFire CUDA library has been found.
    ArrayFire_CUDA_LIBRARIES   - Location of ArrayFire's CUDA library, if found
    ArrayFire_OpenCL_FOUND     - True of the ArrayFire OpenCL library has been found.
    ArrayFire_OpenCL_LIBRARIES - Location of ArrayFire's OpenCL library, if found

Therefore, if you wish to target a specific specific backend, switch
`${ArrayFire_LIBRARIES}` to `${ArrayFire_CPU}` `${ArrayFire_OPENCL}` or
`${ArrayFire_CUDA}` in the `TARGET_LINK_LIBRARIES` command above.

Finally, if you have installed ArrayFire to a non-standard location, CMake can still help
you out. When you execute CMake specify the path to the `ArrayFireConfig*` files that
are found in the `share/ArrayFire` subdirectory of the installation folder.
For example, if ArrayFire were installed locally to `/opt/ArrayFire` then you would
modify the `cmake` command above to contain the following definition:

```
cmake -DArrayFire_DIR=/opt/ArrayFire/share/ArrayFire ...
```

## MakeFiles

Using ArrayFire with Makefiles is almost as easy as CMake, but you will
need to specify paths manually. In your makefile specify the include path to
the directory containing `arrayfire.h`. Typically this will be `-I /usr/include`
or `-I /usr/local/include` if you installed ArrayFire using our installation
instructions.
Then, in your linker line specify the path to ArrayFire using the `-L` option
(typically `-L/usr/lib` or `-L/usr/local/lib` and the specific ArrayFire backend
you wish to use with the `-l` option (i.e. `-lafcpu`, `-lafopencl` or `-lafcuda`
for the CPU, OpenCL and CUDA backends repsectively).

Here is a minimial example MakeFile which uses ArrayFire's CPU backend:

    LIBS=-lafcpu
    LIB_PATHS=/usr/lib
    INCLUDES=-I/usr/include
    CC=g++ $(COMPILER_OPTIONS)
    COMPILER_OPTIONS=-std=c++11 -g

    all: main.cpp Makefile
        $(CC) main.cpp -o test $(INCLUDES) $(LIBS) $(LIB_PATHS)
