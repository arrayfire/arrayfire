Release Notes {#releasenotes}
==============

v3.2.0
=================

Major Updates
-------------

* Added Unified backend
    * Allows switching backends at runtime
    * Read [Unified Backend](\ref unifiedbackend) for more.
* Support for 16-bit integers (\ref s16 and \ref u16)
    * All functions that support 32-bit interger types (\ref s32, \ref u32),
      now also support 16-bit interger types

Function Additions
------------------
* Unified Backend
    * \ref setBackend() - Sets a backend as active
    * \ref getBackendCount() - Gets the number of backends available for use
    * \ref getAvailableBackends() - Returns information about available backends
    * \ref getBackendId() - Gets the backend enum for an array

* Vision
    * \ref homography() - Homography estimation
    * \ref gloh() - GLOH Descriptor for SIFT

* Image Processing
    * \ref loadImageNative() - Load an image as native data without modification
    * \ref saveImageNative() - Save an image without modifying data or type

* Graphics
    * \ref af::Window::plot3() - 3-dimensional line plot
    * \ref af::Window::surface() - 3-dimensional curve plot

* Indexing
    * \ref af_create_indexers()
    * \ref af_set_array_indexer()
    * \ref af_set_seq_indexer()
    * \ref af_set_seq_param_indexer()
    * \ref af_release_indexers()

* CUDA Backend Specific
    * \ref setNativeId() - Set the CUDA device with given native id as active
        * ArrayFire uses a modified order for devices. The native id for a
          device can be retreived using `nvidia-smi`

* OpenCL Backend Specific
    * \ref setDeviceId() - Set the OpenCL device using the `clDeviceId`

Other Improvements
------------------------
* Added \ref c32 and \ref c64 support for \ref isNaN(), \ref isInf() and \ref iszero()
* Added CPU information for `x86` and `x86_64` architectures in CPU backend's \ref info()
* Batch support for \ref approx1() and \ref approx2()
    * Now can be used with gfor as well
* Added \ref s64 and \ref u64 support to:
    * \ref sort() (along with sort index and sort by key)
    * \ref setUnique(), \ref setUnion(), \ref setIntersect()
    * \ref convolve() and \ref fftConvolve()
    * \ref histogram() and \ref histEqual()
    * \ref lookup()
    * \ref mean()
* Added \ref AF_MSG macro

Build Improvements
------------------
* Submodules update is now automatically called if not cloned recursively
* [Fixes for compilation](https://github.com/arrayfire/arrayfire/issues/766) on Visual Studio 2015
* Option to use [fallback to CPU LAPACK](https://github.com/arrayfire/arrayfire/pull/1053)
  for linear algebra functions in case of CUDA 6.5 or older versions.

Bug Fixes
--------------
* Fixed [memory leak](https://github.com/arrayfire/arrayfire/pull/1096) in \ref susan()
* Fixed [failing test](https://github.com/arrayfire/arrayfire/commit/144a2db)
  in \ref lower() and \ref upper() for CUDA compute 53
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1092) in CUDA for indexing out of bounds
* Fixed [dims check](https://github.com/arrayfire/arrayfire/commit/6975da8) in \ref iota()
* Fixed [out-of-bounds access](https://github.com/arrayfire/arrayfire/commit/7fc3856) in \ref sift()
* Fixed [memory allocation](https://github.com/arrayfire/arrayfire/commit/5e88e4a) in \ref fast() OpenCL
* Fixed [memory leak](https://github.com/arrayfire/arrayfire/pull/994) in image I/O functions
* \ref dog() now returns float-point type arrays

Documentation Updates
---------------------
* Improved tutorials documentation
    * More detailed Using on [Linux](\ref using_on_windows), [OSX](\ref using_on_windows),
      [Windows](\ref using_on_windows) pages.
* Added return type information for functions that return different type
  arrays

New Examples
------------
* Graphics
    * [Plot3](\ref plot3.cpp)
    * [Surface](\ref surface.cpp)
* [Shallow Water Equation](\ref swe.cpp)
* [Basic](\ref basic.cpp) as a Unified backend example

Installers
-----------
* All installers now include the Unified backend and corresponding CMake files
* Visual Studio projects include Unified in the Platform Configurations
* Added installer for Jetson TX1
* SIFT and GLOH do not ship with the installers as SIFT is protected by
  patents that do not allow commercial distribution without licensing.

v3.1.3
==============

Bug Fixes
---------

* Fixed [bugs](https://github.com/arrayfire/arrayfire/issues/1042) in various OpenCL kernels without offset additions
* Remove ARCH_32 and ARCH_64 flags
* Fix [missing symbols](https://github.com/arrayfire/arrayfire/issues/1040) when freeimage is not found
* Use CUDA driver version for Windows
* Improvements to SIFT
* Fixed [memory leak](https://github.com/arrayfire/arrayfire/issues/1045) in median
* Fixes for Windows compilation when not using MKL [#1047](https://github.com/arrayfire/arrayfire/issues/1047)
* Fixed for building without LAPACK

Other
-------

* Documentation: Fixed documentation for select and replace
* Documentation: Fixed documentation for af_isnan

v3.1.2
==============

Bug Fixes
---------

* Fixed [bug](https://github.com/arrayfire/arrayfire/commit/4698f12) in assign that was causing test to fail
* Fixed bug in convolve. Frequency condition now depends on kernel size only
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1005) in indexed reductions for complex type in OpenCL backend
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1006) in kernel name generation in ireduce for OpenCL backend
* Fixed non-linear to linear indices in ireduce
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1011) in reductions for small arrays
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1010) in histogram for indexed arrays
* Fixed [compiler error](https://github.com/arrayfire/arrayfire/issues/1015) CPUID for non-compliant devices
* Fixed [failing tests](https://github.com/arrayfire/arrayfire/issues/1008) on i386 platforms
* Add missing AFAPI

Other
-------

* Documentation: Added missing examples and other corrections
* Documentation: Fixed warnings in documentation building
* Installers: Send error messages to log file in OSX Installer

v3.1.1
==============

Installers
-----------

* CUDA backend now depends on CUDA 7.5 toolkit
* OpenCL backend now require OpenCL 1.2 or greater

Bug Fixes
--------------

* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/981) in reductions after indexing
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/976) in indexing when using reverse indices

Build
------

* `cmake` now includes `PKG_CONFIG` in the search path for CBLAS and LAPACKE libraries
* [heston_model.cpp](\ref heston_model.cpp) example now builds with the default ArrayFire cmake files after installation

Other
------

* Fixed bug in [image_editing.cpp](\ref image_editing.cpp)

v3.1.0
==============

Function Additions
------------------
* Computer Vision Functions
    * \ref nearestNeighbour() - Nearest Neighbour with SAD, SSD and SHD distances
    * \ref harris() - Harris Corner Detector
    * \ref susan() - Susan Corner Detector
    * \ref sift() - Scale Invariant Feature Transform (SIFT)
        * Method and apparatus for identifying scale invariant features"
          "in an image and use of same for locating an object in an image,\" David"
          "G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application"
          "filed March 8, 1999. Asignee: The University of British Columbia. For"
          "further details, contact David Lowe (lowe@cs.ubc.ca) or the"
          "University-Industry Liaison Office of the University of British"
          "Columbia.")
        * SIFT is available for compiling but does not ship with ArrayFire
          hosted installers/pre-built libraries
    * \ref dog() -  Difference of Gaussians

* Image Processing Functions
    * \ref ycbcr2rgb() and \ref rgb2ycbcr() - RGB <->YCbCr color space conversion
    * \ref wrap() and \ref unwrap() Wrap and Unwrap
    * \ref sat() - Summed Area Tables
    * \ref loadImageMem() and \ref saveImageMem() - Load and Save images to/from memory
        * \ref af_image_format - Added imageFormat (af_image_format) enum

* Array & Data Handling
    * \ref copy() - Copy
    * array::lock() and array::unlock() - Lock and Unlock
    * \ref select() and \ref replace() - Select and Replace
    * Get array reference count (af_get_data_ref_count)

* Signal Processing
    * \ref fftInPlace() - 1D in place FFT
    * \ref fft2InPlace() - 2D in place FFT
    * \ref fft3InPlace() - 3D in place FFT
    * \ref ifftInPlace() - 1D in place Inverse FFT
    * \ref ifft2InPlace() - 2D in place Inverse FFT
    * \ref ifft3InPlace() - 3D in place Inverse FFT
    * \ref fftR2C() - Real to complex FFT
    * \ref fftC2R() - Complex to Real FFT

* Linear Algebra
    * \ref svd() and \ref svdInPlace() - Singular Value Decomposition

* Other operations
    * \ref sigmoid() - Sigmoid
    * Sum (with option to replace NaN values)
    * Product (with option to replace NaN values)

* Graphics
    * Window::setSize() - Window resizing using Forge API

* Utility
    * Allow users to set print precision (print, af_print_array_gen)
    * \ref saveArray() and \ref readArray() - Stream arrays to binary files
    * \ref toString() - toString function returns the array and data as a string

* CUDA specific functionality
    * \ref getStream() - Returns default CUDA stream ArrayFire uses for the current device
    * \ref getNativeId() - Returns native id of the CUDA device

Improvements
------------
* dot
    * Allow complex inputs with conjugate option
* AF_INTERP_LOWER interpolation
    * For resize, rotate and transform based functions
* 64-bit integer support
    * For reductions, random, iota, range, diff1, diff2, accum, join, shift
      and tile
* convolve
    * Support for non-overlapping batched convolutions
* Complex Arrays
    * Fix binary ops on complex inputs of mixed types
    * Complex type support for exp
* tile
    * Performance improvements by using JIT when possible.
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
    * Fix [bug](https://github.com/arrayfire/arrayfire/issues/901) in rank
    * Fix default values for scale throwing exception
    * Fix conjg raising exception on real input
    * Fix bug when using conjugate transpose for vector input
    * Fix issue with const input for array_proxy::get()
* CPU Backend
    * Fix randn generating same sequence for multiple calls
    * Fix setSeed for randu
    * Fix casting to and from complex
    * Check NULL values when allocating memory
    * Fix [offset issue](https://github.com/arrayfire/arrayfire/issues/923) for CPU element-wise operations

New Examples
------------
* Match Template
* Susan
* Heston Model (contributed by Michael Nowotny)

Installer
----------
* Fixed bug in automatic detection of ArrayFire when using with CMake in Windows
* The Linux libraries are now compiled with static version of FreeImage

Known Issues
------------
* OpenBlas can cause issues with QR factorization in CPU backend
* FreeImage older than 3.10 can cause issues with loadImageMem and
  saveImageMem
* OpenCL backend issues on OSX
    * AMD GPUs not supported because of driver issues
    * Intel CPUs not supported
    * Linear algebra functions do not work on Intel GPUs.
* Stability and correctness issues with open source OpenCL implementations such as Beignet, GalliumCompute.

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
