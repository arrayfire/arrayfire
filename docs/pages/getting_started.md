Getting Started {#gettingstarted}
===============

[TOC]

Basic data types and arithmetic {#gettingstarted_datatypes}
===============================

There is one generic [array](\ref af::array) container object while the
underlying data may be one of various basic types:
* [b8](\ref b8) 8-bit boolean values (`bool`)
* [f32](\ref f32) real single-precision (`float`)
* [c32](\ref c32) complex single-precision (`cfloat`)
* [s32](\ref s32) 32-bit signed integer (`int`)
* [u32](\ref u32) 32-bit unsigned integer (`unsigned`)
* [f64](\ref f64) real double-precision (`double`)
* [c64](\ref c64) complex double-precision (`cdouble`)
* [s64](\ref s64) 64-bit signed integer (`intl`)
* [u64](\ref u64) 64-bit unsigned integer (`uintl`)


Older devices may not support double precision operations

You can [generate](\ref data_mat) matrices out on the device.  The
default underlying datatype is [f32](\ref f32) (`float`) unless
otherwise specified.  Some examples:

\snippet test/getting_started.cpp ex_getting_started_gen

You can also initialize values from a host array:

\snippet test/getting_started.cpp ex_getting_started_init

You can print the contents of an array or expression:

\snippet test/getting_started.cpp ex_getting_started_print

You can access the dimensions of a matrix using a [dim4](\ref af::dim4) object
or directly via [dims()](\ref af::array::dims) and [numdims()](\ref af::array::numdims)

\snippet test/getting_started.cpp ex_getting_started_dims

You can query properties about an array:

\snippet test/getting_started.cpp ex_getting_started_prop

There are [hundreds of functions](\ref arith_mat) for element-wise arithmetic:

\snippet test/getting_started.cpp ex_getting_started_arith

You can initialize a matrix from either a host or device pointer:

\snippet test/getting_started.cpp ex_getting_started_dev_ptr

You can get both device and host side pointers to the underlying
data with [device()](\ref af::array::device) and [host()](\ref af::array::host)

\snippet test/getting_started.cpp ex_getting_started_ptr

You can pull the scalar value from the first element of an array back to the CPU
with [scalar()](\ref af::array::scalar).

\snippet test/getting_started.cpp ex_getting_started_scalar

Integer support includes bitwise operations as well as the
standard alogirthms like [sort, sum, minmax](\ref vector_mat), [indexing](\ref indexing)
(see [more](\ref integer.cpp)).

\snippet test/getting_started.cpp ex_getting_started_bit

Several platform-independent constants are available: [Pi](\ref af::Pi),
[NaN](\ref af::NaN), and [Inf](\ref af::Inf)
When these variable names conflict with macros in the standard header
files or variables in scope, then reference them with their full namespace,
e.g. af::NaN

\snippet test/getting_started.cpp ex_getting_started_constants

C API {#gettingstarted_c_api}
========================

Basic Example
-------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    #include <arrayfire.h>
    // Generate random data and add all the values (Ignore error codes)
    int main(void)
    {
        // generate random values
        int n = 10000;
        af_array a;
        af_randu(&a, n);

        // sum all the values
        double real, imag;
        af_sum_all(&real, &imag, n, a);

        printf("sum: %g\n", sum);
        return 0;
    }
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
