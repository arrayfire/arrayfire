Using ArrayFire on Linux {#using_on_linux}
=====

Once you have [installed](\ref installing) ArrayFire on your system, the next
thing to do is set up your build system. On Linux, you can create ArrayFire
projects using almost any editor, compiler, or build system. The only
requirements are that you include the ArrayFire header directories and link with
the ArrayFire library you intend to use i.e. CUDA, OpenCL, CPU, or Unified
backends.

## The big picture  {#big-picture-linux}

On Linux, we recommend installing ArrayFire to `/opt/arrayfire` directory. The
installer will populate files in the following sub-directories:

    include/arrayfire.h         - Primary ArrayFire include file
    include/af/*.h              - Additional include files
    lib/libaf*                  - CPU, CUDA, oneAPI and OpenCL libraries (.a, .so)
    lib/libforge*               - Visualization library
    lib/libcu*                  - CUDA backend dependencies
    lib/libOpenCL.so            - OpenCL ICD Loader library
    share/ArrayFire/cmake/*     - CMake config (find) scripts
    share/ArrayFire/examples/*  - All ArrayFire examples

Because ArrayFire follows standard installation practices, you can use basically
any build system to create and compile projects that use ArrayFire. Among the
many possible build systems on Linux we suggest using ArrayFire with either
CMake or Makefiles with CMake being our preferred build system.

## Prerequisite software

To build ArrayFire projects you will need a compiler

#### Fedora, Centos and Redhat

Install EPEL repo (not required for Fedora)

```
yum install epel-release
yum update
```

Install build dependencies

```
yum install gcc gcc-c++ cmake3 make
```

#### Debian and its derivatives

Install common dependencies

```
apt install build-essential cmake cmake-curses-gui
```

## CMake

We recommend that the CMake build system be used to create ArrayFire projects.
As [discussed above](#big-picture-linux), ArrayFire ships with a series of CMake
scripts to make finding and using our library easy.

First create a file called `CMakeLists.txt` in your project directory:

    cd your-project-directory
    touch CMakeLists.txt

and populate it with the following code:

    find_package(ArrayFire)
    add_executable(<my_executable> [list your source files here])

    # To use Unified backend, do the following.
    # Unified backend lets you choose the backend at runtime
    target_link_libraries(<my_executable> ArrayFire::af)

where `my_executable` is the name of the executable you wish to create. See the
[CMake documentation](https://cmake.org/documentation/) for more information on
how to use CMake. To link with a specific backend directly, replace the
`ArrayFire::af` with the following for their respective backends.

* `ArrayFire::afcpu` for CPU backend.
* `ArrayFire::afcuda` for CUDA backend.
* `ArrayFire::afoneapi` for oneAPI backend.
* `ArrayFire::afopencl` for OpenCL backend.

Next we need to instruct CMake to create build instructions and then compile. We
suggest using CMake's out-of-source build functionality to keep your build and
source files cleanly separated. To do this open the CMake GUI.

    cd your-project-directory
    mkdir build
    cd build
    cmake ..
    make

*NOTE:* If you have installed ArrayFire to a non-standard location, CMake can
still help you out. When you execute CMake specify the path to ArrayFire
installation root as `ArrayFire_DIR` variable.

For example, if ArrayFire were installed locally to `/home/user/ArrayFire` then
you would modify the `cmake` command above to contain the following definition:

    cmake -DArrayFire_DIR=/home/user/ArrayFire ..

You can also specify this information in the `ccmake` command-line interface.

## Makefiles

Building ArrayFire projects with Makefiles is fairly similar to CMake except you
must specify all paths and libraries manually.

As with any `make` project, you need to specify the include path to the
directory containing `arrayfire.h` file. This should be `-I
/opt/arrayfire/include` if you followed our installation instructions.

Similarly, you will need to specify the path to the ArrayFire library using the
`-L` option (e.g. `-L/opt/arrayfire/lib`) followed by the specific ArrayFire
library you wish to use using the `-l` option (for example `-lafcpu`,
`-lafopencl`, `-lafoneapi`, `-lafcuda`, or `-laf` for the CPU, OpenCL, oneAPI
and CUDA, and unified backends, respectively.

Here is a minimal example Makefile which uses ArrayFire's CPU backend:

    LIBS=-lafcpu
    LIB_PATHS=-L/opt/arrayfire/lib
    INCLUDES=-I/opt/arrayfire/include
    CC=g++ $(COMPILER_OPTIONS)
    COMPILER_OPTIONS=-std=c++11 -g

    all: main.cpp Makefile
        $(CC) main.cpp -o test $(INCLUDES) $(LIBS) $(LIB_PATHS)
