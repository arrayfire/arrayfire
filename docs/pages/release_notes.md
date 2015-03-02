Release Notes {#releasenotes}
==============

Major Updates
-------------

* Support for Mac OS X
* New Language support ([available on github](https://github.com/arrayfire))
    * [ArrayFire for R!](https://github.com/arrayfire/arrayfire_r)
    * [ArrayFire for Fortran](https://github.com/arrayfire/arrayfire_fortran)<sup>#</sup>
    * [ArrayFire for Java](https://github.com/arrayfire/arrayfire_java)
* [ArrayFire Extras on Github](https://github.com/arrayfire)
    * [Rolling updates to examples](https://github.com/arrayfire/arrayfire_examples)
    * [OpenGL interop example](https://github.com/arrayfire/arrayfire_opengl_interop)

<sup>#</sup> ArrayFire for Fortan has been removed from the installed library,
but can be downloaded from the ArrayFire Extras GitHub page linked above.

Function Additions
------------------
* Image transformation (warp affine) functions
    * transform()
        * affine and inverse affine transform of an image
    * translate()
        * translate an image using affine transforms
    * scale()
        * scale an image using affine transforms
    * skew()
        * skew an image using affine transforms
* Coordinate transformation (homogeneous transformation) functions
    * transform_coords()
        * Upto 3D homogeneous transformations
    * rotate_coords()
        * rotation matrix wrapper for homogeneous coordinate transformation

API Changes
---------------------
* rotate()
    * Added optional 4th parameter `recenter`
* af_filter removed

Feature Improvements
--------------------
* approx1() and approx2() now support nearest interpolation
* rotate(), resize(), convolve() work for stack of images(3D) and gfor
* new indexing functions to add support for 4-th dimension gfor
* host pinned memory support for OpenCL

Bug Fixes
---------
* loadimage()
    * fixed bug with grayscale image reading
* gaussiankernel()
    * fixed computation of gaussian kernel values
* resize()
    * fixed resize in OpenCL
* medfilt()
    * fixed to handle more data types
* rotate()
    * fixed rounding issue on Tahiti GPUs
* approx1()
    * fixed for gfor use
* flip()
    * fixed launch configuration
* indexing now works on intel GPUs

Performance Improvements
------------------------
* rotate() rewritten to improve performance
