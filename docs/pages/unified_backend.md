Unified Backend {#unifiedbackend}
==========

[TOC]

# Introduction

The Unified backend was introduced in ArrayFire with version 3.2.
While this is not an independent backend, it allows the user to switch between
the different ArrayFire backends (CPU, CUDA and OpenCL) at runtime.

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
priority of backends is __CUDA -> OpenCL -> CPU__

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
    ArrayFire v3.2.0 (CPU, 64-bit Linux, build fc7630f)
    [0] Intel: Intel(R) Core(TM) i7-4770K CPU @ 3.50GHz Max threads(8)
    af::randu(5, 4)
    [5 4 1 1]
        0.0000     0.2190     0.3835     0.5297
        0.1315     0.0470     0.5194     0.6711
        0.7556     0.6789     0.8310     0.0077
        0.4587     0.6793     0.0346     0.3834
        0.5328     0.9347     0.0535     0.0668

    Trying CUDA Backend
    ArrayFire v3.2.0 (CUDA, 64-bit Linux, build fc7630f)
    Platform: CUDA Toolkit 7.5, Driver: 355.11
    [0] Quadro K5000, 4093 MB, CUDA Compute 3.0
    af::randu(5, 4)
    [5 4 1 1]
        0.7402     0.4464     0.7762     0.2920
        0.9210     0.6673     0.2948     0.3194
        0.0390     0.1099     0.7140     0.8109
        0.9690     0.4702     0.3585     0.1541
        0.9251     0.5132     0.6814     0.4452

    Trying OpenCL Backend
    ArrayFire v3.2.0 (OpenCL, 64-bit Linux, build fc7630f)
    [0] NVIDIA  : Quadro K5000
    -1- INTEL   : Intel(R) Core(TM) i7-4770K CPU @ 3.50GHz
    af::randu(5, 4)
    [5 4 1 1]
        0.4107     0.0081     0.6600     0.1046
        0.8224     0.3775     0.0764     0.8827
        0.9518     0.3027     0.0901     0.1647
        0.1794     0.6456     0.5933     0.8060
        0.4198     0.5591     0.1098     0.5938

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
