Release Notes {#releasenotes}
==============

Major Updates
-------------

* ArrayFire is now open source
* New backend: CPU fallback for systems without GPUs
* A new and improved C api
* Support for 64 bit integers

Function Additions
------------------
* Data generation functions
    * range()
    * iota()

* Computer Vision Algorithms
    * fast()
        * FAST feature detector
    * orb()
        * ORB A feature descriptor extractor

* Image Processing
    * convolve1(), convolve2(), convolve3()
        * Specialized versions of convolve() to enable better batch support
    * matchTemplate()
        * Match a kernel in an image
    * sobel()
        * Get sobel gradients of an image

* Matrix Multiply
    * matmulNT(), matmulTN(), matmulTT()
        * Specialized versions of matmul() for transposed inputs

* Other functions
    * lookup() - lookup indices from a table

Deprecated Function APIs
------------------------

Deprecated APIs are in af/compatible.h

* devicecount() changed to getDeviceCount()
* deviceset() changed to setDevice()
* deviceget() changed to getDevice()
* loadimage() changed to loadImage()
* saveimage() changed to saveImage()
* gaussiankernel() changed to gaussianKernel()

API Changes
---------------------
* `print` is now af_print()

Performance Improvements
------------------------
* Improvements across the board for OpenCL backend
