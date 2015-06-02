Overview {#mainpage}
========

[TOC]

## About ArrayFire

ArrayFire is a high performance software library for parallel computing with an easy-to-use API. Its array based function set makes parallel programming more accessible.

## Installing ArrayFire

You can install ArrayFire using either a binary installer for Windows, OSX,
or Linux or download it from source:

* [Binary installers for Windows, OSX, and Linux](\ref installing)
* [Build from source](https://github.com/arrayfire/arrayfire)

## Easy to use

The [array](\ref construct_mat) object is beautifully simple.

Array-based notation effectively expresses computational algorithms in
readable math-resembling notation. You _do not_ need expertise in
parallel programming to use ArrayFire.

A few lines of ArrayFire code
accomplishes what can take 100s of complicated lines in CUDA or OpenCL
kernels.

## ArrayFire is extensive!

#### Support for multiple domains

ArrayFire contains [hundreds of functions](\ref arrayfire_func) across various domains including:
- [Vector Algorithms](\ref vector_mat)
- [Image Processing](\ref image_mat)
- [Computer Vision](\ref cv_mat)
- [Signal Processing](\ref signal_mat)
- [Linear Algebra](\ref linalg_mat)
- [Statistics](\ref stats_mat)
- and more.

Each function is hand-tuned by ArrayFire
developers with all possible low-level optimizations.

#### Support for various data types and sizes

ArrayFire operates on common [data shapes and sizes](\ref indexing),
including vectors, matrices, volumes, and

It supports common [data types](\ref gettingstarted_datatypes),
including single and double precision floating
point values, complex numbers, booleans, and 32-bit signed and
unsigned integers.

#### Extending ArrayFire

ArrayFire can be used as a stand-alone application or integrated with
existing CUDA or OpenCL code. All ArrayFire `arrays` can be
interchanged with other CUDA or OpenCL data structures.

## Code once, run anywhere!

With support for x86, ARM, CUDA, and OpenCL devices, ArrayFire supports for a comprehensive list of devices.

Each ArrayFire installation comes with:
 - a CUDA version (named 'libafcuda') for [NVIDIA
 GPUs](https://developer.nvidia.com/cuda-gpus),
 - an OpenCL version (named 'libafopencl') for [OpenCL devices](http://www.khronos.org/conformance/adopters/conformant-products#opencl)
 - a CPU version (named 'libafcpu') to fall back to when CUDA or OpenCL devices are not available.

## ArrayFire is highly efficient

#### Vectorized and Batched Operations

ArrayFire supports batched operations on N-dimensional arrays.
Batch operations in ArrayFire are run in parallel ensuring an optimal usage of your CUDA or OpenCL device.

You can get the best performance out of ArrayFire using [vectorization techniques]().

ArrayFire can also execute loop iterations in parallel with
[the gfor function](\ref gfor).

#### Just in Time compilation

ArrayFire performs run-time analysis of your code to increase
arithmetic intensity and memory throughput, while avoiding unnecessary
temporary allocations. It has an awesome internal JIT compiler to make
optimizations for you.

Read more about how [ArrayFire JIT](http://arrayfire.com/performance-of-arrayfire-jit-code-generation/) can improve the performance in your application.

## Simple Example

Here's a live example to let you see ArrayFire code. You create [arrays](\ref
construct_mat) which reside on CUDA or OpenCL devices. Then you can use
[ArrayFire functions](modules.htm) on those [arrays](\ref construct_mat).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// sample 40 million points on the GPU
array x = randu(20e6), y = randu(20e6);
array dist = sqrt(x * x + y * y);

// pi is ratio of how many fell in the unit circle
float num_inside = sum<float>(dist < 1);
float pi = 4.0 * num_inside / 20e6;
af_print(pi);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Product Support

#### Free Community Options

* [ArrayFire mailing list](https://groups.google.com/forum/#!forum/arrayfire-users) (recommended)
* [StackOverflow](http://stackoverflow.com/questions/tagged/arrayfire)

#### Premium Support

* Phone Support - available for purchase ([request a quote](mailto:sales@arrayfire.com))

#### Contact Us

* If you need to contact us, visit our
[contact us page](http://arrayfire.com/company/#contact).

#### Email

* Engineering: technical@arrayfire.com
* Sales: sales@arrayfire.com
