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
* [s64](\ref c64) complex double-precision (`intl`)
* [u64](\ref c64) complex double-precision (`uintl`)


Older devices may not support double precision operations

You can [generate](\ref gen) matrices out on the device.  The
default underlying datatype is [f32](\ref f32) (`float`) unless
otherwise specified.  Some examples:

\democode{
af_print(constant(0, 3));
af_print(constant(1, 3, 2, f64));
af_print(randu(1, 4));
af_print(randn(2, 2));
af_print(identity(3, 3));
af_print(randu(2, 1, c32));
}

You can also initialize values from a host array:

\democode{
float hA[] = {0, 1, 2, 3, 4, 5};
array A(2, 3, hA);   // 2x3 matrix of single-precision
af_print(A);            // Note: Fortran storage order (column major)
}

You can print the contents of an array or expression:

\democode{
array a = randu(2, 2);
array b = constant(1, 2, 1);
af_print(a);
af_print(b);
af_print(a.col(0) + b + .4);
}

You can access the dimensions of a matrix using a [dim4](\ref af::dim4) object
or directly via [dims()](\ref af::array::dims) and [numdims()](\ref af::array::numdims)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array a = randu(4,5,2);
printf("numdims(a)  %d\n",  a.numdims()); // 3

dim4 dims = a.dims();
printf("dims = [%d %d]\n", dims[0], dims[1]); // 4,5
printf("dims = [%d %d]\n", a.dims(0), a.dims(1)); // 4,5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can query properties about an array:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
printf("underlying type: %d\n", a.type());
printf("is complex? %d    is real? %d\n", a.iscomplex(), a.isreal());
printf("is vector? %d  column? %d  row? %d\n", a.isvector(), a.iscolumn(), a.isrow());
printf("empty? %d  total elements: %d  bytes: %zu\n", a.isempty(), a.elements(), a.bytes());
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are [hundreds of functions](\ref arith) for element-wise arithmetic:

\democode{
array R = randu(3, 3);
af_print(constant(1, 3, 3) + complex(sin(R)));  // will be c32

// rescale complex values to unit circle
array a = randn(5, c32);
af_print(a / abs(a));

// calculate L2 norm of vectors
array X = randn(3, 4);
af_print(sqrt(sum(pow(X, 2))));     // norm of every column vector
af_print(sqrt(sum(pow(X, 2), 0)));  // same as above
af_print(sqrt(sum(pow(X, 2), 1)));  // norm of every row vector
}

You can initialize a matrix from either a host or device pointer:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
float host_ptr[] = {0,1,2,3,4,5}; // column-major order (like Fortran)
array a(2, 3, host_ptr); // f32 matrix of size 2-by-3 from host data

float *device_ptr;
cudaMalloc((void**)&device_ptr, 6*sizeof(float));
cudaMemcpy(device_ptr, host_ptr, 6*sizeof(float), cudaMemcpyHostToDevice);
array b(2,3, host_ptr, afDevice); // Note: afDevice (default: afHost)
// do not call cudaFree(device_ptr) -- it is freed when 'b' is destructed.

// create complex data
cuComplex ha[] = { {0,1}, {2,3}, {4,5} }; // { {real,imaginary}, {real,imag}, .. }
array a(3,ha); // 3x1 column vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can get both device and host side pointers to the underlying
data with [device()](\ref af::array::device) and [host()](\ref af::array::host)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array a = randu(3, f32);
float *host_a = a.host<float>();        // must call array::free() later
printf("host_a[2] = %g\n", host_a[2]);  // last element
array::free(host_a);

float *device_a = a.device<float>();    // no need to free this
float value;
cudaMemcpy(&value, device_a + 2, sizeof(float), cudaMemcpyDeviceToHost);
printf("device_a[2] = %g\n", value);
a.unlock(); // unlock to allow garbage collection if necessary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can pull the scalar value from the first element of an array back to the CPU
with [scalar()](\ref af::array::scalar).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array a = randu(3);
float val = a.scalar<float>();
printf("scalar value: %g\n", val);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integer support includes bitwise operations as well as the
standard sort(), [min/max](\ref minmax_mat), [indexing](\ref gettingstarted_indexing)
(see [more](\ref examples/getting_started/integer.cpp)).

\democode{
int h_A[] = {1, 1, 0, 0, 4, 0, 0, 2, 0};
int h_B[] = {1, 0, 1, 0, 1, 0, 1, 1, 1};
array A = array(3, 3, h_A), B = array(3, 3, h_B);
af_print(A); af_print(B);

array A_and_B = A & B; af_print(A_and_B);
array  A_or_B = A | B; af_print(A_or_B);
array A_xor_B = A ^ B; af_print(A_xor_B);
}

Several platform-independent constants are available: [Pi](\ref af::Pi),
[NaN](\ref af::NaN), [Inf](\ref af::Inf), [i](\ref af::i).
When these variable names conflict with macros in the standard header
files or variables in scope, then reference them with their full namespace,
e.g. af::NaN

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = randu(5,5);
A(A > .5) = af::NaN;

array x = randu(20e6), y = randu(20e6);
double pi_est = 4 * sum<float>(hypot(x,y) < 1) / 20e6;
printf("estimation error: %g\n", fabs(Pi - pi_est));
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
