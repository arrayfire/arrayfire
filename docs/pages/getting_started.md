Getting Started {#gettingstarted}
===============

[TOC]

Basic data types and arithmetic {#gettingstarted_datatypes}
===============================

There is one generic [array](\ref af::array) container object while the
underlying data may be one of various basic types:
* [b8](\ref af::b8) 8-bit boolean values (`bool`)
* [f32](\ref af::f32) real single-precision (`float`)
* [c32](\ref af::c32) complex single-precision (`cfloat`)
* [s32](\ref af::s32) 32-bit signed integer (`int`)
* [u32](\ref af::u32) 32-bit unsigned integer (`unsigned`)
* [f64](\ref af::f64) real double-precision (`double`)
* [c64](\ref af::c64) complex double-precision (`cdouble`)

Older devices may not support double precision operations

You can [generate](\ref gen) matrices out on the device.  The
default underlying datatype is [f32](\ref af::f32) (`float`) unless
otherwise specified.  Some examples:

\democode{
print(constant(0, 3));
print(constant(1, 3, 2, f64));
print(randu(1, 4));
print(randn(2, 2));
print(identity(3, 3));
print(randu(2, 1, c32));
print(rand(2, 4, u32));
}

You can also initialize values from a host array:

\democode{
float hA[] = {0, 1, 2, 3, 4, 5};
array A(2, 3, hA);   // 2x3 matrix of single-precision
print(A);            // Note: Fortran storage order (column major)
}

You can print the contents of an array or expression:

\democode{
array a = randu(2, 2);
array b = constant(1, 2, 1);
print(a);
print(b);
print(a.col(0) + b + .4);
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
print(constant(1, 3, 3) + complex(sin(R)));  // will be c32

// rescale complex values to unit circle
array a = randn(5, c32);
print(a / abs(a));

// calculate L2 norm of vectors
array X = randn(3, 4);
print(sqrt(sum(pow(X, 2))));     // norm of every column vector
print(sqrt(sum(pow(X, 2), 0)));  // same as above
print(sqrt(sum(pow(X, 2), 1)));  // norm of every row vector
}

\note By default \ref blas_mat "A*B" implements elementwise multiply;
however, you can \ref blas_mat "toggle this" to be matrix
multiply to favor linear algebra, otherwise simply use \ref matmul().

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
print(A); print(B);

array A_and_B = A & B; print(A_and_B);
array  A_or_B = A | B; print(A_or_B);
array A_xor_B = A ^ B; print(A_xor_B);
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

Matrix Manipulation {#gettingstarted_matrixmanipulation}
===================

Many different kinds of [matrix manipulation routines](\ref manip_mat) are available:
* tile() to repeat a matrix along dimensions
* join() to concatenate two matrices along a dimension
* [array()](\ref af::array) to adjust the dimensions of an array
* [transpose](\ref af::array::T) a matrix or vector

tile() allows you to repeat a matrix along specified
dimensions, effectively 'tiling' the matrix.  Please note that the
dimensions passed in indicate the number of times to replicate the
matrix in each dimension, not the final dimensions of the matrix.

\democode{
float h[] = {1, 2, 3, 4};
array small = array(2, 2, h); // 2x2 matrix
print(small);
array large = tile(small, 2, 3);  // produces 8x12 matrix: (2*2)x(2*3)
}

join() allows you to joining two matrices together.  Matrix
dimensions must match along every dimension except the dimension
of joining (dimensions are 0-indexed). For example, a 2x3 matrix
can be joined with a 2x4 matrix along dimension 1, but not along
dimension 0 since {3,4} don`t match up.

\democode{
float hA[] = { 1, 2, 3, 4, 5, 6 };
float hB[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
array A = array(3, 2, hA);
array B = array(3, 3, hB);

print(join(1, A, B)); // 3x5 matrix
// array result = join(0, A, B); // fail: dimension mismatch
}

Construct a regular mesh grid from vectors `x` and `y`. For example, a
mesh grid of the vectors {1,2,3,4} and {5,6} would result in two matrices:

\democode{
float hx[] = {1, 2, 3, 4};
float hy[] = {5, 6};

array x = array(4, hx);
array y = array(2, hy);

print(tile(x, 1, 2));
print(tile(y.T(), 4, 1));
}

[array()](\ref af::array) can be used to create a (shallow) copy of a matrix
with different dimensions.  The number of elements must remain the same as
the original array.

\democode{
int hA[] = {1, 2, 3, 4, 5, 6};
array A = array(3, 2, hA);

print(array(A, 2, 3)); // 2x3 matrix
print(array(A, 6, 1)); // 6x1 column vector

// print(array(A, 2, 2)); // fail: wrong number of elements
// print(array(A, 8, 8)); // fail: wrong number of elements
}

The [T()](\ref af::array::T) and [H()](\ref af::array::H) methods can be
used to form the [matrix or vector transpose](\ref af::array::T) .

\democode{
array x = randu(2, 2, f64);
print(x.T());  // transpose (real)

array c = randu(2, 2, c64);
print(c.T());  // transpose (complex)
print(c.H());  // Hermitian (conjugate) transpose
}

Indexing {#gettingstarted_indexing}
========

There are several ways of referencing values.  ArrayFire uses
parenthesis for subscripted referencing instead of the traditional
square bracket notation.  Indexing is zero-based, i.e. the first
element is at index zero (<tt>A(0)</tt>).  Indexing can be done
with mixtures of:
* integer scalars
* [seq()](\ref af::seq) representing a linear sequence
* [end](\ref af::end) representing the last element of a dimension
* [span](\ref af::span) representing the entire dimension
* [row(i)](\ref af::array::row) or [col(i)](\ref af::array::col) specifying a single row/column
* [rows(first,last)](\ref af::array::rows) or [cols(first,last)](\ref af::array::cols)
 specifying a span of rows or columns

See \ref gettingstarted_indexing for the full listing.

\democode{
array A = array(seq(1,9), 3, 3);
print(A);

print(A(0));    // first element
print(A(0,1));  // first row, second column

print(A(end));   // last element
print(A(-1));    // also last element
print(A(end-1)); // second-to-last element

print(A(1,span));       // second row
print(A.row(end));      // last row
print(A.cols(1,end));   // all but first column

float b_host[] = {0,1,2,3,4,5,6,7,8,9};
array b(10, 1, b_host);
print(b(seq(3)));
print(b(seq(1,7)));
print(b(seq(1,2,7)));
print(b(seq(0,2,end)));
}

You can set values in an array:

\democode{
array A = constant(0, 3, 3);

// setting entries to a constant
A(span) = 4;        // fill entire array
print(A);

A.row(0) = -1;      // first row
print(A);

A(seq(3)) = 3.1415; // first three elements
print(A);

// copy in another matrix
array B = constant(1, 4, 4, f64);
B.row(0) = randu(1, 4, f32); // set a row to random values (also upcast)
}


Use one array to reference into another.

\democode{
float h_inds[] = {0, 4, 2, 1}; // zero-based indexing
array inds(1, 4, h_inds);
print(inds);

array B = randu(1, 4);
print(B);

array c = B(inds);        // get
print(c);

B(inds) = -1;             // set to scalar
B(inds) = constant(0, 4); // zero indices
print(B);
}

Convolutions {#gettingstarted_convolutions}
============

The \ref convolve() is the single entry point for all image and signal convolution:
* vectors (1D), images (2D), volumes (3D)
* separable and expanded convolution
* kernels in either device or host memory

\ref convolve() with two inputs performs \a N dimensional
convolution, where \a N is the highest input dimension:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    array image  = randu(10,10);
    array kernel = constant(1,3,3) / 9; // average within 3x3 window
    print(convolve(image,kernel)); // 10x10 blurred image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However if the kernel is small and is on the host, it`s faster to
use it directly from the host pointer instead of pushing it to
device first:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    array signal = randu(5000,1);
    float host_filter[] = {1, 0, -1};
    unsigned filter_dims[] = {3};
    convolve(signal,
             1,         // number of filter dimensions
             filter_dims, // filter dimensions
             host_filter);// filter inside host memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, a 2D filter kernel is considered _separable_,
meaning it can be decomposed into two orthogonal vectors.
Convolving with those individual vectors is almost always faster.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    // 5x5 derivative with separable kernels
    float h_dx[] = {1.f/12, -8.f/12, 0, 8.f/12, -1.f/12}; // five point stencil
    float h_spread[] = {1.f/5, 1.f/5, 1.f/5, 1.f/5, 1.f/5};
    array dx = array(5,1,h_dx);
    array spread = array(1,5,h_spread);
    array kernel = dx * spread; // 5x5 derivative kernel

    array image = randu(640,480);
    convolve(image, kernel); // derivative of image going down columns

    // equivalent and faster version:
    convolve(dx,spread,image);

    // also supports passing host pointers:
    convolve(5,h_dx, 5,h_spread, image);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running the [convolve.cpp example](\ref examples/getting_started/convolve.cpp)
shows nearly a 3x difference between the separable and
non-separable cases:

    arrayfire/examples/getting_started $ ./convolve
    full 2D convolution:         0.00156023
    separable, device pointers:  0.000595222
    separable, host pointers:    0.000590385

You can also produce different parts of the convolution with the

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    convolve(randu(3,1), randu(5,1))  // 3x1 output
    convolve(randu(5,1), randu(3,1))  // 5x1 output
    convolve(randu(3,1), randu(5,1), true)  // 7x1 output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\sa \ref examples/getting_started/convolve.cpp

Device Pointer Interface {#gettingstarted_devicepointer}
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
        float *d_values;
        af_randu_S(&d_values,  n);

        // sum all the values
        float sum;
        af_sum_vector_S(&sum, n, d_values);
        printf("sum: %g\n", sum);
        return 0;
    }
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the batch parameter

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    #include <arrayfire.h>
    // Performs 10 convolutions in parallel (ignore error checking)
    int main(void)
    {
        // generate random values
        int m = 10000; // Length of Signal
        int n = 500;   // Length of Kernel
        int k = 10;    // Number of Signals to convolve

        float *d_signal, *d_filter, *d_result;

        // Generate 'k' random signals each of length m
        af_randu_S(&d_signal, m * k);

        // Generate one random kernel to convolve the signals with
        af_randu_S(&d_filter, n * 1);

        // Allocate space for result
        af_malloc(&d_result, (m + n - 1) * k * sizeof(float));

        // Perform the convolutions
        af_conv_SS(d_result,       // output
                   m, d_signal, k, // Signal size, pointer, number of signals
                   n, d_filter, 1, // Kernel size, pointer, number of kernels
                   1);             // (FULL: 1, SAME: 0, VALID: -1)
        return 0;
    }
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Timing Your Code {#gettingstarted_timing}
================

timer() : A platform-independent timer with microsecond accuracy:
* [timer::start()](\ref af::timer::start) starts a timer

* [timer::start()](\ref af::timer::stop) seconds since last \ref af::timer::start "start"

* \ref af::timer::stop(af::timer start) "timer::start(timer start)" seconds since 'start'

Example: single timer

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   // start timer
   timer::start();
   // run your code
   printf("elapsed seconds: %g\n", timer::stop());
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: multiple timers

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   // start timers
   timer start1 = timer::start();
   timer start2 = timer::start();
   // run some code
   printf("elapsed seconds: %g\n", timer::stop(start1));
   // run more code
   printf("elapsed seconds: %g\n", timer::stop(start2));
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accurate and reliable measurement of performance involves several factors:
* Executing enough iterations to achieve peak performance.
* Executing enough repetitions to amortize any overhead from system timers.

To take care of much of this boilerplate, [timeit](\ref af::timeit) provides
accurate and reliable estimates of both CPU or GPU code.

Here`s a stripped down example of
[Monte-Carlo estimation of PI](\ref examples/benchmarks/pi.cpp) making use
of [timeit](\ref af::timeit).  Notice how it expects a `void` function pointer.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <stdio.h>
#include <arrayfire.h>
using namespace af;

void pi_function() {
  int n = 20e6; // 20 million random samples
  array x = randu(n,f32), y = randu(n,f32);
  // how many fell inside unit circle?
  float pi = 4.0 * sum<float>(sqrt(x*x + y*y)) < 1) / n;
}

int main() {
  printf("pi_function took %g seconds\n", timeit(pi_function));
  return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This produces:

	pi_function took 0.007252 seconds
	(test machine: Core i7 920 @ 2.67GHz with a Tesla C2070)
