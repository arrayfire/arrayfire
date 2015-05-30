Getting Started {#gettingstarted}
===============

[TOC]

# Supported data types {#gettingstarted_datatypes}

There is one generic [array](\ref af::array) container object while the
underlying data may be one of various [basic types](\ref af::af_dtype):

* [b8](\ref b8) 8-bit boolean values (`bool`)
* [f32](\ref f32) real single-precision (`float`)
* [c32](\ref c32) complex single-precision (`cfloat`)
* [s32](\ref s32) 32-bit signed integer (`int`)
* [u32](\ref u32) 32-bit unsigned integer (`unsigned`)
* [f64](\ref f64) real double-precision (`double`)
* [c64](\ref c64) complex double-precision (`cdouble`)
* [s64](\ref s64) 64-bit signed integer (`intl`)
* [u64](\ref u64) 64-bit unsigned integer (`uintl`)

Older devices may not support double precision operations.

# Creating an populating an ArrayFire array {#getting_started_af_arrays}

ArrayFire [array](\ref af::array)s always exist on the device. They
may be populated with data using an ArrayFire function, or filled with data
found on the host. For example:

\snippet test/getting_started.cpp ex_getting_started_gen

A complete list of ArrayFire functions that automatically generate data
on the device may be found on the [functions to create arrays](\ref data_mat)
page. The default data type for arrays is [f32](\ref f32) (a
32-bit floating point number) unless specified otherwise.

ArrayFire arrays may also be populated from data found on the host.
For example:

\snippet test/getting_started.cpp ex_getting_started_init

ArrayFire also supports array initialization from a device pointer.
For example ArrayFire can be populated directly by a call to `cudaMemcpy`

\snippet test/getting_started.cpp ex_getting_started_dev_ptr

# ArrayFire array contents, dimentions, and properties {#getting_started_array_properties}

The [af_print](\ref af::af_print) function can be used to print arrays that
have already been generated or an expression involving arrays:

\snippet test/getting_started.cpp ex_getting_started_print

ArrayFire provides several convenient methods for accessing the dimensions.
You may use either a [dim4](\ref af::dim4) object or access the dimensions
directly using the [dims()](\ref af::array::dims) and
[numdims()](\ref af::array::numdims) functions:

\snippet test/getting_started.cpp ex_getting_started_dims

Arrays also provide functions to determine their properties including:

\snippet test/getting_started.cpp ex_getting_started_prop

# Writing mathematical expressions in ArrayFire {#getting_started_writing_math}

Most of ArrayFire's functions operate on an element-wise basis.
This means that function like `c[i] = a[i] + b[i]` could simply be written
as `c = a + b`.
ArrayFire has an intelligent runtime JIT compliation engine which converts
array expressions into the smallest number of OpenCL/CUDA kernels.
This "kernel fusion" technology not only decreases the number of kernel calls,
but, more importantly, avoids extraneous global memory operations.
Our JIT functionality extends across C/C++ function boundaries and only ends
when a non-JIT function is encountered or a synchronization operation is
explicitly called by the code.

ArrayFire has [hundreds of functions](\ref arith_mat) for element-wise
arithmetic. Here are a few examples:

\snippet test/getting_started.cpp ex_getting_started_arith

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

Like all functions in ArrayFire, indexing is also executed in parallel on
the OpenCL/CUDA device.
Because of this, indexing becomes part of a JIT operation and is accomplished
using parentheses instead of square brackets (i.e. as `A(0)` instead of `A[0]`).
To index `af::array`s you may use one or a combination of the following functions:

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
and device()](\ref af::array::device) functions.
The `host` function *copies* the data from the device and makes it available
in a C-style array on the host.
The `device` function returns a pointer to device memory for interoperability
with external CUDA/OpenCL kernels.
For example, here is how we can interact with both OpenCL and CUDA:

\snippet test/getting_started.cpp ex_getting_started_ptr

ArrayFire also provides several helper functions for creating `af::array`s from
OpenCL `cl_mem` references and `cl::Buffer` objects. See the `include/af/opencl.h`
file for further information.

Lastly, if you want only the first value from an `af::array` you can use
get it using the [scalar()](\ref af::array::scalar) function:

\snippet test/getting_started.cpp ex_getting_started_scalar

# Bitwise operators {#getting_started_bitwise_operators}

In addition to supporting standard mathematical functions, `af::array`s
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
        int n = 10000;
        af_array a;
        af_randu(&a, n);

        // sum all the values
        float result;
        af_sum_all(&result, a, 0);

        printf("sum: %g\n", sum);
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
