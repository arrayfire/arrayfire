Overview {#mainpage}
========

[TOC]

## About ArrayFire

ArrayFire is a high performance software library for parallel computing with
an easy-to-use API. Its array based function set makes parallel programming
more accessible.

## Installing ArrayFire

Install ArrayFire using either a binary installer for Windows, OSX, or Linux
or download it from source:

* [Binary installers for Windows, OSX, and Linux](\ref installing)
* [Build from source](https://github.com/arrayfire/arrayfire)

## Easy to use

The [array](\ref af::array) object is beautifully simple.

Array-based notation effectively expresses computational algorithms in
readable math-resembling notation. Expertise in parallel programming _is not_
required to use ArrayFire.

A few lines of ArrayFire code accomplishes what can take 100s of complicated
lines in CUDA, oneAPI, or OpenCL kernels.

## ArrayFire is extensive!

#### Support for multiple domains

ArrayFire contains [hundreds of functions](\ref arrayfire_func) across various
domains including:
- [Vector Algorithms](\ref vector_mat)
- [Image Processing](\ref image_mat)
- [Computer Vision](\ref cv_mat)
- [Signal Processing](\ref signal_mat)
- [Linear Algebra](\ref linalg_mat)
- [Statistics](\ref stats_mat)
- and more.

Each function is hand-tuned by ArrayFire developers with all possible
low-level optimizations.

#### Support for various data types and sizes

ArrayFire operates on common [data shapes and sizes](\ref indexing), including
vectors, matrices, volumes, and

It supports common [data types](\ref gettingstarted_datatypes), including
single and double precision floating point values, complex numbers, booleans,
and 32-bit signed and unsigned integers.

#### Extending ArrayFire

ArrayFire can be used as a stand-alone application or integrated with existing
CUDA, oneAPI, or OpenCL code.

## Code once, run anywhere!

With support for x86, ARM, CUDA, oneAPI, and OpenCL devices, ArrayFire
supports for a comprehensive list of devices.

Each ArrayFire installation comes with:
- a CUDA backend (named 'libafcuda') for [NVIDIA
  GPUs](https://developer.nvidia.com/cuda-gpus),
- a oneAPI backend (named 'libafoneapi') for [oneAPI
  devices](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html),
- an OpenCL backend (named 'libafopencl') for [OpenCL
  devices](http://www.khronos.org/conformance/adopters/conformant-products#opencl),
- a CPU backend (named 'libafcpu') to fall back to when CUDA, oneAPI, or
  OpenCL devices are unavailable.

## ArrayFire is highly efficient

#### Vectorized and Batched Operations

ArrayFire supports batched operations on N-dimensional arrays. Batch
operations in ArrayFire are run in parallel ensuring an optimal usage of CUDA,
oneAPI, or OpenCL devices.

Best performance with ArrayFire is achieved using
[vectorization techniques](\ref vectorization).

ArrayFire can also execute loop iterations in parallel with
[the gfor function](\ref gfor).

#### Just in Time compilation

ArrayFire performs run-time analysis of code to increase arithmetic intensity
and memory throughput, while avoiding unnecessary temporary allocations. It
has an awesome internal JIT compiler to make important optimizations.

Read more about how [ArrayFire JIT](\ref jit).  can improve the performance in
your application.

## Simple Example

Here is an example of ArrayFire code that performs a Monte Carlo estimation of
PI.

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

* [ArrayFire mailing
  list](https://groups.google.com/forum/#!forum/arrayfire-users) (recommended)
* [StackOverflow](http://stackoverflow.com/questions/tagged/arrayfire)

#### Premium Support

* Phone Support - available for purchase ([request a
  quote](mailto:sales@arrayfire.com))

#### Contact Us

* If you need to contact us, visit our [contact us
  page](http://arrayfire.com/company/#contact).

#### Email

* Engineering: technical@arrayfire.com
* Sales: sales@arrayfire.com

## Citations and Acknowledgements

If you redistribute ArrayFire, please follow the terms established in <a
href="https://github.com/arrayfire/arrayfire/blob/master/LICENSE">the
license</a>. If you wish to cite ArrayFire in an academic publication, please
use the following reference:

Formatted:

    Yalamanchili, P., Arshad, U., Mohammed, Z., Garigipati, P., Entschev, P.,
    Kloppenborg, B., Malcolm, James and Melonakos, J. (2015).
    ArrayFire - A high performance software library for parallel computing with an
    easy-to-use API. Atlanta: AccelerEyes. Retrieved from https://github.com/arrayfire/arrayfire

BibTeX:

    @misc{Yalamanchili2015,
    abstract = {ArrayFire is a high performance software library for parallel computing with an easy-to-use API. Its array based function set makes parallel programming simple. ArrayFire's multiple backends (CUDA, OpenCL and native CPU) make it platform independent and highly portable. A few lines of code in ArrayFire can replace dozens of lines of parallel computing code, saving you valuable time and lowering development costs.},
    address = {Atlanta},
    author = {Yalamanchili, Pavan and Arshad, Umar and Mohammed, Zakiuddin and Garigipati, Pradeep and Entschev, Peter and Kloppenborg, Brian and Malcolm, James and Melonakos, John},
    publisher = {AccelerEyes},
    title = {{ArrayFire - A high performance software library for parallel computing with an easy-to-use API}},
    url = {https://github.com/arrayfire/arrayfire},
    year = {2015}
    }

ArrayFire development is funded by AccelerEyes LLC (dba ArrayFire) and several
third parties, please see the list of <a
href="https://github.com/arrayfire/arrayfire/blob/master/ACKNOWLEDGEMENTS.md">acknowledgements</a>.
