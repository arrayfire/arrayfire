Overview {#mainpage}
========

[TOC]

Make your code run faster
-------------------------

ArrayFire will make your code run as fast as possible. It beats
efforts to manually write CUDA or OpenCL kernels. It beats compiler
optimizations. It beats other libraries. ArrayFire is the best way to
accelerate your code.

ArrayFire developers are amazingly talented at accelerating code;
that's all we do - ever!

With minimal effort
-------------------

The [array](\ref construct) object is beautifully simple. It's fun to
use!

Array-based notation effectively expresses computational algorithms in
readable math-resembling notation. You _do not_ need expertise in
parallel programming to use ArrayFire. A few lines of ArrayFire code
accomplishes what can take 100s of complicated lines in CUDA or OpenCL
kernels.

Save yourself from verbose templates, ineffective and complicated
compiler directives, and time-wasting low-level development. Arrays
are the best possible way to accelerate your code.

On CUDA or OpenCL devices (e.g. GPUs, CPUs, APUs, FPGAs)
--------------------------------------------------------

ArrayFire supports CUDA and OpenCL capable devices. Each ArrayFire
installation comes with a CUDA version (named 'libafcu') for [NVIDIA
GPUs](https://developer.nvidia.com/cuda-gpus) and an OpenCL version
(named 'libafcl') for [OpenCL
devices](http://www.khronos.org/conformance/adopters/conformant-products#opencl).

You can easily switch between CUDA or OpenCL with ArrayFire, without
changing your code.

For common science, engineering, and financial functions
------------------------------------------------------------

ArrayFire contains [hundreds of functions](modules.htm) for matrix
arithmetic, signal processing, linear algebra, statistics, image
processing, and more. Each function is hand-tuned by ArrayFire
developers with all possible low-level optimizations.

For common data shapes, sizes, and types
--------------------------------------------

ArrayFire operates on common [data shapes and sizes](\ref gettingstarted_indexing),
including vectors, matrices, volumes, and
N-dimensional arrays. It supports common [data types](\ref gettingstarted_datatypes),
including single and double precision floating
point values, complex numbers, booleans, and 32-bit signed and
unsigned integers.

With available integration into CUDA or OpenCL kernel code
---------------------------------------------------------------

ArrayFire can be used as a stand-alone application or integrated with
existing CUDA or OpenCL code. All ArrayFire `arrays` can be
interchanged with other CUDA or OpenCL data structures.

With awesome automatic optimizations
------------------------------------

ArrayFire performs run-time analysis of your code to increase
arithmetic intensity and memory throughput, while avoiding unnecessary
temporary allocations. It has an awesome internal JIT compiler to make
optimizations for you.

With parallel for-loops
-----------------------

ArrayFire can also execute loop iterations in parallel with
[the gfor function](\ref gfor).

With multi-GPU or multi-device scalability
------------------------------------------

ArrayFire supports easy [multi-GPU or multi-device](\ref device_mat)
scaling.

Simple Example {#simpleexample}
==============

Here's a live example to let you see ArrayFire code. You create [arrays](\ref
construct) which reside on CUDA or OpenCL devices. Then you can use
[ArrayFire functions](modules.htm) on those [arrays](\ref construct).

<div class="AF_div" style="height: 160px"><pre>
// sample 40 million points on the GPU
array x = randu(20e6), y = randu(20e6);
array dist = sqrt(x * x + y * y);
|
// pi is ratio of how many fell in the unit circle
array pi = 4.0 * sum(dist < 1) / 20e6;
print(pi);</pre></div>



Product Support {#support}
===============

Free Community Options
----------------------

* [ArrayFire Forums](http://forums.accelereyes.com) (recommended)
* [StackOverflow](http://stackoverflow.com/questions/tagged/arrayfire)

Premium Support
---------------

* Phone Support - available for purchase ([request a quote](mailto:sales@arrayfire.com))

Contact Us
----------

* If you need to contact us, visit our
[contact us page](http://arrayfire.com/company/#contact).
