Unified Backend {#unifiedbackend}
==========

[TOC]

# Introduction

The Unified backend was introduced in ArrayFire with version 3.2.
While this is not an independent backend, it allows the user to switch between
the different ArrayFire backends (CPU, CUDA, oneAPI and OpenCL) at runtime.

# Compiling with Unified

The steps to compile with the unified backend are the same as compiling with
any of the other backends.
The only change being that the executable needs to be linked with the __af__
library (`libaf.so` (Linux), `libaf.dylib` (OSX), `af.lib` (Windows)).

Check the Using with [Linux](\ref using_on_linux), [OSX](\ref using_on_osx),
[Windows](\ref using_on_windows) for more details.

To use with CMake, use the __ArrayFire_Unified_LIBRARIES__ variable.

# Using the Unified Backend

The Unified backend will try to dynamically load the backend libraries. The
priority of backends is __CUDA -> oneAPI -> OpenCL -> CPU__

The most important aspect to note here is that all the libraries the ArrayFire
libs depend on need to be in the environment paths

* `LD_LIBRARY_PATH` -> Linux, Unix, OSX
* `DYLD_LIBRARY_PATH` -> OSX
* `PATH` -> Windows

If any of the libs are missing, then the library will fail to load and the
backend will be marked as unavailable.

Optionally, The ArrayFire libs may be present in `AF_PATH` or `AF_BUILD_PATH`
environment variables if the path is not in the system paths. These are
treated as fallback paths in case the files are not found in the system paths.
However, all the other upstream libraries for ArrayFire libs must be present
in the system path variables shown above.

# Switching Backends

The af_backend enum stores the possible backends.
To select a backend, call the af::setBackend function as shown below.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
af::setBackend(AF_BACKEND_CUDA);    // Sets CUDA as current backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the count of the number of backends available (the number of `libaf*`
backend libraries loaded successfully), call the af::getBackendCount function.

# Example

This example is shortened form of [basic.cpp](\ref unified/basic.cpp).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
#include <arrayfire.h>

void testBackend()
{
    af::info();
    af_print(af::randu(5, 4));
}

int main()
{
    try {
        printf("Trying CPU Backend\n");
        af::setBackend(AF_BACKEND_CPU);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying CPU backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("Trying oneAPI Backend\n");
        af::setBackend(AF_BACKEND_ONEAPI);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying oneAPI backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("Trying CUDA Backend\n");
        af::setBackend(AF_BACKEND_CUDA);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying CUDA backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("Trying OpenCL Backend\n");
        af::setBackend(AF_BACKEND_OPENCL);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying OpenCL backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This output would be:

    Trying CPU Backend
    ArrayFire v3.9.0 (CPU, 64-bit Linux, build 23ee0650e)
    [0] AMD: AMD Ryzen Threadripper PRO 3955WX 16-Cores     af::randu(5, 4)
    [5 4 1 1]
        0.6010     0.5497     0.1583     0.3636
        0.0278     0.2864     0.3712     0.4165
        0.9806     0.3410     0.3543     0.5814
        0.2126     0.7509     0.6450     0.8962
        0.0655     0.4105     0.9675     0.3712

    Trying oneAPI Backend
    ArrayFire v3.9.0 (oneAPI, 64-bit Linux, build 23ee0650e)
    [0] Intel(R) OpenCL: AMD Ryzen Threadripper PRO 3955WX 16-Cores     , 128650 MB (fp64)
    af::randu(5, 4)
    [5 4 1 1]
        0.6010     0.5497     0.1583     0.3636
        0.0278     0.2864     0.3712     0.4165
        0.9806     0.3410     0.3543     0.5814
        0.2126     0.7509     0.6450     0.8962
        0.0655     0.4105     0.9675     0.3712

    Trying CUDA Backend
    ArrayFire v3.9.0 (CUDA, 64-bit Linux, build 23ee0650e)
    Platform: CUDA Runtime 12.2, Driver: 535.104.05
    [0] NVIDIA RTX A5500, 22721 MB, CUDA Compute 8.6
    -1- NVIDIA RTX A5500, 22719 MB, CUDA Compute 8.6
    af::randu(5, 4)
    [5 4 1 1]
        0.6010     0.5497     0.1583     0.3636
        0.0278     0.2864     0.3712     0.4165
        0.9806     0.3410     0.3543     0.5814
        0.2126     0.7509     0.6450     0.8962
        0.0655     0.4105     0.9675     0.3712

    Trying OpenCL Backend
    ArrayFire v3.9.0 (OpenCL, 64-bit Linux, build 23ee0650e)
    [0] NVIDIA: NVIDIA RTX A5500, 22720 MB
    -1- NVIDIA: NVIDIA RTX A5500, 22718 MB
    -2- Intel(R) FPGA Emulation Platform for OpenCL(TM): Intel(R) FPGA Emulation Device, 128650 MB
    -3- INTEL: AMD Ryzen Threadripper PRO 3955WX 16-Cores     , 128650 MB
    af::randu(5, 4)
    [5 4 1 1]
        0.6010     0.5497     0.1583     0.3636
        0.0278     0.2864     0.3712     0.4165
        0.9806     0.3410     0.3543     0.5814
        0.2126     0.7509     0.6450     0.8962
        0.0655     0.4105     0.9675     0.3712


# Dos and Don'ts

It is very easy to run into exceptions if you are not careful with the
switching of backends.

### Don't: Do not use arrays between different backends

ArrayFire checks the input arrays to functions for mismatches with the active
backend. If an array created on one backend, but used when another backend is
set to active, an exception with code 503 (`AF_ERR_ARR_BKND_MISMATCH`) is
thrown.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
#include <arrayfire.h>

int main()
{
    try {
        af::setBackend(AF_BACKEND_CUDA);
        af::array A = af::randu(5, 5);

        af::setBackend(AF_BACKEND_OPENCL);
        af::array B = af::constant(10, 5, 5);
        af::array C = af::matmul(A, B);     // This will throw an exception

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }

    return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Do: Use a naming scheme to track arrays and backends

We recommend that you use a technique to track the arrays on the backends. One
suggested technique would be to use a suffix of `_cpu`, `_cuda`, `_opencl`
with the array names. So an array created on the CUDA backend would be named
`myarray_cuda`.

If you have not used the af::setBackend function anywhere in your code, then
you do not have to worry about this as all the arrays will be created on the
same default backend.

### Don't: Do not use custom kernels (CUDA/OpenCL) with the Unified backend

This is another area that is a no go when using the Unified backend. It not
recommended that you use custom kernels with unified backend. This is mainly
becuase the Unified backend is meant to be ultra portable and should use only
ArrayFire and native CPU code.
