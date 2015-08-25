Release Notes {#releasenotes}
==============

v3.1.0
==============

Function Additions
------------------
* Computer Vision Functions
    * Nearest Neighbour with SAD, SSD and SHD distances (nearestNeighbour)
    * Harris Corner Detector (harris)
    * Susan Corner Detector (susan)
    * Scale Invariant Feature Transform (SIFT)
        * Method and apparatus for identifying scale invariant features"
          "in an image and use of same for locating an object in an image,\" David"
          "G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application"
          "filed March 8, 1999. Asignee: The University of British Columbia. For"
          "further details, contact David Lowe (lowe@cs.ubc.ca) or the"
          "University-Industry Liaison Office of the University of British"
          "Columbia.")
        * SIFT is available for compiling but does not ship with ArrayFire
          hosted installers/pre-built libraries
    * Difference of Gaussians (dog)

* Image Processing Functions
    * RGB <->YCbCr color space conversion (ycbcr2rgb, rgb2ycbcr)
    * Wrap and Unwrap (wrap, unwrap)
    * Summed Area Tables (sat)
    * Load and Save images to/from memory (loadImageMem, saveImageMem)
        * Add imageFormat (af_image_format) enum

* Utility
    * Allow users to set print precision (print, af_print_array_gen)
    * Stream arrays to binary files (saveArray, readArray)
    * toString function returns the array and data as a string

* Array & Data Handling
    * Copy (copy)
    * Lock and Unlock (array::lock, array::unlock)
    * Select and Replace (select, replace)
    * Get array reference count (af_get_data_ref_count)

* Other operations
    * SVD Decomposition (svd)
    * FFT
        * Support for in place FFT
    * Sigmoid (sigmoid)
    * Sum (with option to replace NaN values)
    * Product (with option to replace NaN values)

* Graphics
    * Window resizing using Forge API (Window::setSize)

Improvements
------------
* dot
    * Allow complex inputs with conjugate option
* AF_INTERP_LOWER interpolation
    * For resize, rotate and transform based functions
* 64-bit integer support
    * For reductions, random, iota, range, diff1, diff2, accum, join, shift
      and tile
* sum and product
    * Support for NaN value substitution
* convolve
    * Support for non-overlapping batched convolutions
* Complex Arrays
    * Fix binary ops on complex inputs of mixed types
    * Complex type support for exp
* FFT
    * Support for r2c and c2r FFT
    * Support for in place FFT
* tile
    * is now a JIT function
* Add AF_API_VERSION macro
    * Allows disabling of API to maintain consistency with previous versions
* Other Performance Improvements
    * Use reference counting to reduce unnecessary copies
* CPU Backend
    * Device properties for CPU
    * Improved performance when all buffers are indexed linearly
* CUDA Backend
    * Use streams in CUDA (no longer using default stream)
    * Using async cudaMem ops
    * Add 64-bit integer support for JIT functions
    * Performance improvements for CUDA JIT for non-linear 3D and 4D arrays
* OpenCL Backend
    * Improve compilation times for OpenCL backend
    * Performance improvements for non-linear JIT kernels on OpenCL
    * Improved shared memory load/store in many OpenCL kernels (PR 933)
    * Using cl.hpp v1.2.7

Bug Fixes
---------
* Common
    * Fix compatibility of c32/c64 arrays when operating with scalars
    * Fix median for all values of an array
    * Fix double free issue when indexing (30cbbc7)
    * Fix bug in rank
    * Fix default values for scale throwing exception
    * Fix conjg raising exception on real input
    * Fix bug when using conjugate transpose for vector input
    * Fix issue with const input for array_proxy::get()
* CPU Backend
    * Fix randn generating same sequence for multiple calls
    * Fix setSeed for randu
    * Fix casting to and from complex
    * Check NULL values when allocating memory
    * Fix offset issue for CPU element-wise operations

New Examples
------------
* Match Template
* Susan
* Heston Model (contributed by Michael Nowotny)

Distribution Changes
--------------------
* Fixed automatic detection of ArrayFire when using with CMake in the Windows
  Installer
* Compiling ArrayFire with FreeImage as a static library for Linux x86
  installers

Known Issues
------------
* OpenBlas can cause issues with QR
* FreeImage older than 3.10 can cause issues with loadImageMem and
  saveImageMem

v3.0.2
==============

Bug Fixes
--------------

* Added missing symbols from the compatible API
* Fixed a bug affecting corner rows and elements in \ref grad()
* Fixed linear interpolation bugs affecting large images in the following:
    - \ref approx1()
    - \ref approx2()
    - \ref resize()
    - \ref rotate()
    - \ref scale()
    - \ref skew()
    - \ref transform()

Documentation
-----------------

* Added missing documentation for \ref constant()
* Added missing documentation for `array::scalar()`
* Added supported input types for functions in `arith.h`

v3.0.1
==============

Bug Fixes
--------------

* Fixed header to work in Visual Studio 2015
* Fixed a bug in batched mode for FFT based convolutions
* Fixed graphics issues on OSX
* Fixed various bugs in visualization functions

Other improvements
---------------

* Improved fractal example
* New OSX installer
* Improved Windows installer
  * Default install path has been changed
* Fixed bug in machine learning examples

<br>

v3.0.0
=================

Major Updates
-------------

* ArrayFire is now open source
* Major changes to the visualization library
* Introducing handle based C API
* New backend: CPU fallback available for systems without GPUs
* Dense linear algebra functions available for all backends
* Support for 64 bit integers

Function Additions
------------------
* Data generation functions
    * range()
    * iota()

* Computer Vision Algorithms
    * features()
        * A data structure to hold features
    * fast()
        * FAST feature detector
    * orb()
        * ORB A feature descriptor extractor

* Image Processing
    * convolve1(), convolve2(), convolve3()
        * Specialized versions of convolve() to enable better batch support
    * fftconvolve1(), fftconvolve2(), fftconvolve3()
        * Convolutions in frequency domain to support larger kernel sizes
    * dft(), idft()
        * Unified functions for calling multi dimensional ffts.
    * matchTemplate()
        * Match a kernel in an image
    * sobel()
        * Get sobel gradients of an image
    * rgb2hsv(), hsv2rgb(), rgb2gray(), gray2rgb()
        * Explicit function calls to colorspace conversions
    * erode3d(), dilate3d()
        * Explicit erode and dilate calls for image morphing

* Linear Algebra
    * matmulNT(), matmulTN(), matmulTT()
        * Specialized versions of matmul() for transposed inputs
    * luInPlace(), choleskyInPlace(), qrInPlace()
        * In place factorizations to improve memory requirements
    * solveLU()
        * Specialized solve routines to improve performance
    * OpenCL backend now Linear Algebra functions

* Other functions
    * lookup() - lookup indices from a table
    * batchFunc() - helper function to perform batch operations

* Visualization functions
    * Support for multiple windows
    * window.hist()
        * Visualize the output of the histogram

* C API
    * Removed old pointer based C API
    * Introducing handle base C API
    * Just In Time compilation available in C API
    * C API has feature parity with C++ API
    * bessel functions removed
    * cross product functions removed
    * Kronecker product functions removed

Performance Improvements
------------------------
* Improvements across the board for OpenCL backend

API Changes
---------------------
* `print` is now af_print()
* seq(): The step parameter is now the third input
    * seq(start, step, end) changed to seq(start, end, step)
* gfor(): The iterator now needs to be seq()

Deprecated Function APIs
------------------------
Deprecated APIs are in af/compatible.h

* devicecount() changed to getDeviceCount()
* deviceset() changed to setDevice()
* deviceget() changed to getDevice()
* loadimage() changed to loadImage()
* saveimage() changed to saveImage()
* gaussiankernel() changed to gaussianKernel()
* alltrue() changed to allTrue()
* anytrue() changed to anyTrue()
* setunique() changed to setUnique()
* setunion() changed to setUnion()
* setintersect() changed to setIntersect()
* histequal() changed to histEqual()
* colorspace() changed to colorSpace()
* filter() deprecated. Use convolve1() and convolve2()
* mul() changed to product()
* deviceprop() changed to deviceProp()

Known Issues
----------------------
* OpenCL backend issues on OSX
    * AMD GPUs not supported because of driver issues
    * Intel CPUs not supported
    * Linear algebra functions do not work on Intel GPUs.
* Stability and correctness issues with open source OpenCL implementations such as Beignet, GalliumCompute.
