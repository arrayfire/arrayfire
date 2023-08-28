Getting Started {#gettingstarted}
===============

[TOC]

# Introduction

ArrayFire is a high performance software library for parallel computing with
an easy-to-use API. ArrayFire abstracts away much of the details of
programming parallel architectures by providing a high-level container object,
the [array](\ref af::array), that represents data stored on a CPU, GPU, FPGA,
or other type of accelerator. This abstraction permits developers to write
massively parallel applications in a high-level language where they need
not be concerned about low-level optimizations that are frequently required to
achieve high throughput on most parallel architectures.

# Supported data types {#gettingstarted_datatypes}

ArrayFire provides one generic container object, the [array](\ref af::array)
on which functions and mathematical operations are performed. The `array`
can represent one of many different [basic data types](\ref af_dtype):

* [f32](\ref f32) real single-precision (`float`)
* [c32](\ref c32) complex single-precision (`cfloat`)
* [f64](\ref f64) real double-precision (`double`)
* [c64](\ref c64) complex double-precision (`cdouble`)
* [f16](\ref f16) real half-precision (`half_float::half`)
* [b8](\ref b8) 8-bit boolean values (`bool`)
* [s32](\ref s32) 32-bit signed integer (`int`)
* [u32](\ref u32) 32-bit unsigned integer (`unsigned`)
* [u8](\ref u8) 8-bit unsigned values (`unsigned char`)
* [s64](\ref s64) 64-bit signed integer (`intl`)
* [u64](\ref u64) 64-bit unsigned integer (`uintl`)
* [s16](\ref s16) 16-bit signed integer (`short`)
* [u16](\ref u16) 16-bit unsigned integer (`unsigned short`)

Most of these data types are supported on all modern GPUs; however, some
older devices may lack support for double precision arrays. In this case,
a runtime error will be generated when the array is constructed.

If not specified otherwise, `array`s are created as single precision floating
point numbers (`f32`).

# Creating and populating an ArrayFire array {#getting_started_af_arrays}

ArrayFire [array](\ref af::array)s represent memory stored on the device.
As such, creation and population of an array will consume memory on the device
which cannot freed until the `array` object goes out of scope. As device memory
allocation can be expensive, ArrayFire also includes a memory manager which
will re-use device memory whenever possible.

Arrays can be created using one of the [array constructors](\ref af::array).
Below we show how to create 1D, 2D, and 3D arrays with uninitialized values:

\snippet test/getting_started.cpp ex_getting_started_constructors

However, uninitialized memory is likely not useful in your application.
ArrayFire provides several convenient functions for creating arrays that contain
pre-populated values including constants, uniform random numbers, uniform
normally distributed numbers, and the identity matrix:

\snippet test/getting_started.cpp ex_getting_started_gen

A complete list of ArrayFire functions that automatically generate data
on the device may be found on the [functions to create arrays](\ref data_mat)
page. As stated above, the default data type for arrays is [f32](\ref f32) (a
32-bit floating point number) unless specified otherwise.

ArrayFire `array`s may also be populated from data found on the host.
For example:

\snippet test/getting_started.cpp ex_getting_started_init

ArrayFire also supports array initialization from memory already on the GPU.
For example, with CUDA one can populate an `array` directly using a call
to `cudaMemcpy`:

\snippet test/getting_started.cpp ex_getting_started_dev_ptr

Similar functionality exists for OpenCL too. If you wish to intermingle
ArrayFire with CUDA or OpenCL code, we suggest you consult the
[CUDA interoperability](\ref interop_cuda) or
[OpenCL interoperability](\ref interop_opencl) pages for detailed instructions.

# ArrayFire array contents, dimensions, and properties {#getting_started_array_properties}

ArrayFire provides several functions to determine various aspects of arrays.
This includes functions to print the contents, query the dimensions, and
determine various other aspects of arrays.

The [af_print](\ref af_print) function can be used to print arrays that
have already been generated or any expression involving arrays:

\snippet test/getting_started.cpp ex_getting_started_print

The dimensions of an array may be determined using either a
[dim4](\ref af::dim4) object or by accessing the dimensions directly using the
[dims()](\ref af::array::dims) and [numdims()](\ref af::array::numdims)
functions:

\snippet test/getting_started.cpp ex_getting_started_dims

In addition to dimensions, arrays also carry several properties including
methods to determine the underlying type and size (in bytes). You can even
determine whether the array is empty, real/complex, a row/column, or a scalar
or a vector:

\snippet test/getting_started.cpp ex_getting_started_prop

For further information on these capabilities, we suggest you consult the
full documentation on the [array](\ref af::array).

# Writing mathematical expressions in ArrayFire {#getting_started_writing_math}

ArrayFire features an intelligent Just-In-Time (JIT) compilation engine that
converts expressions using arrays into the smallest number of CUDA/OpenCL
kernels. For most operations on arrays, ArrayFire functions like a vector library.
That means that an element-wise operation, like `c[i] = a[i] + b[i]` in C,
would be written more concisely without indexing, like `c = a + b`.
When there are multiple expressions involving arrays, ArrayFire's JIT engine
will merge them together. This "kernel fusion" technology not only decreases
the number of kernel calls, but, more importantly, avoids extraneous global
memory operations.
Our JIT functionality extends across C/C++ function boundaries and only ends
when a non-JIT function is encountered or a synchronization operation is
explicitly called by the code.

ArrayFire provides [hundreds of functions](\ref arith_mat) for element-wise
operations. All of the standard operators (e.g. +,-,\*,/) are supported
as are most transcendental functions (sin, cos, log, sqrt, etc.).
Here are a few examples:

\snippet test/getting_started.cpp ex_getting_started_arith

To see the complete list of functions please consult the documentation on
[mathematical](\ref mathfunc_mat), [linear algebra](\ref linalg_mat),
[signal processing](\ref signal_mat), and [statistics](\ref stats_mat).

# Mathematical constants {#getting_started_constants}

ArrayFire contains several platform-independent constants, like
[Pi](\ref af::Pi), [NaN](\ref af::NaN), and [Inf](\ref af::Inf).
If ArrayFire does not have a constant you need, you can create your own
using the [af::constant](\ref af::constant) array constructor.

Constants can be used in all of ArrayFire's functions. Below we demonstrate
their use in element selection and a mathematical expression:

\snippet test/getting_started.cpp ex_getting_started_constants

Please note that our constants may, at times, conflict with macro definitions
in standard header files. When this occurs, please refer to our constants
using the `af::` namespace.

# Indexing {#getting_started_indexing}

Like all functions in ArrayFire, indexing is also executed in parallel on the
OpenCL/CUDA devices. Because of this, indexing becomes part of a JIT operation
and is accomplished using parentheses instead of square brackets (i.e. as `A(0)`
instead of `A[0]`). To index `af::array`s you may use one or a combination of
the following functions:

* integer scalars
* [seq()](\ref af::seq) representing a linear sequence
* [end](\ref af::end) representing the last element of a dimension
* [span](\ref af::span) representing the entire dimension
* [row(i)](\ref af::array::row) or [col(i)](\ref af::array::col) specifying a single row/column
* [rows(first,last)](\ref af::array::rows) or [cols(first,last)](\ref af::array::cols)
 specifying a span of rows or columns

Please see the [indexing page](\ref indexing) for several examples of how to
use these functions.

# Getting access to ArrayFire array memory on the host and device {#getting_started_memory_access}

Memory in `af::array`s may be accessed using the [host()](\ref af::array::host)
and [device()](\ref af::array::device) functions.
The `host` function *copies* the data from the device and makes it available
in a C-style array on the host. As such, it is up to the developer to manage
any memory returned by `host`.
The `device` function returns a pointer/reference to device memory for
interoperability with external CUDA/OpenCL kernels. As this memory belongs to
ArrayFire, the programmer should not attempt to free/deallocate the pointer.
For example, here is how we can interact with both OpenCL and CUDA:

\snippet test/getting_started.cpp ex_getting_started_ptr

ArrayFire also provides several helper functions for creating `af::array`s from
OpenCL `cl_mem` references and `cl::Buffer` objects. See the `include/af/opencl.h`
file for further information.

Lastly, if you want only the first value from an `af::array` you can use
get it using the [scalar()](\ref af::array::scalar) function:

\snippet test/getting_started.cpp ex_getting_started_scalar

# Bitwise operators {#getting_started_bitwise_operators}

In addition to supporting standard mathematical functions, arrays
that contain integer data types also support bitwise operators including
and, or, and shift:

\snippet test/getting_started.cpp ex_getting_started_bit

# Using the ArrayFire API in C and C++ {#gettingstarted_api_usage}

The ArrayFire API is wrapped into a unified C/C++ header. To use the library
simply include the `arrayfire.h` header file and start coding!

## Sample using the C API

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    #include <arrayfire.h>
    // Generate random data and sum and print the result
    int main(void)
    {
        // generate random values
        af_array a;
        int n_dims = 1;
        dim_t dims[] = {10000};
        af_randu(&a, n_dims, dims, f32);

        // sum all the values
        double result;
        af_sum_all(&result, 0, a);
        printf("sum: %g\n", result);

        return 0;
    }
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Sample using the C++ API

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    #include <arrayfire.h>
    // Generate random data, sum and print the result.
    int main(void)
    {
        // Generate 10,000 random values
        af::array a = af::randu(10000);

        // Sum the values and copy the result to the CPU:
        double sum = af::sum<float>(a);

        printf("sum: %g\n", sum);
        return 0;
    }
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# What to read next? {#getting_started_next_steps}

Now that you have a general introduction to ArrayFire, where do you go from
here? In particular you might find these documents useful

* [Building an ArrayFire program on Linux](\ref using_on_linux)
* [Building an Arrayfire program on Windows](\ref using_on_windows)
* [Timing ArrayFire code](\ref timing)



# Where to go for help? {#getting_started_help}

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:technical@arrayfire.com>
