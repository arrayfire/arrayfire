/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

/**

\defgroup arrayfire_func Complete List of ArrayFire Functions
@{
@}

\defgroup graphics_func Graphics
@{
@}

@defgroup func_categories ArrayFire Functions by Category
@{

   @defgroup array_mat Functions to create and modify Arrays
   @{

      Array constructors, random number generation, transpose, indexing, etc.

      @defgroup construct_mat Constructors of array class
      Construct an array object

      @defgroup method_mat Methods of array class
      Get information about the array object

      @defgroup device_mat Managing devices in ArrayFire
      getting device pointer, allocating and freeing memory

      @defgroup data_mat Functions to create arrays.
      constant, random, range, etc.

      @defgroup index_mat Indexing operation on arrays
      Access sub regions of an array object

      @defgroup manip_mat Move and Reorder array content
      reorder, transpose, flip, join, tile, etc.

      @defgroup helper_mat Helper functions for arrays
      iszero, isInf, isNan, etc.
   @}

   @defgroup mathfunc_mat Mathematical functions
   @{

      Functions from standard math library

      @defgroup arith_mat Arithmetic operations
      +, -, *, /, >>, <<

      @defgroup logic_mat  Logical operations
      &&, ||, |, &, <, >, <=, >=, ==, !

      @defgroup numeric_mat Numeric functions
      floor, round, min, max, etc.

      @defgroup trig_mat Trigonometric functions
      sin, cos, tan, etc.

      @defgroup explog_mat Exponential and logarithmic functions
      exp, log, expm1, log1p, etc.

      @defgroup hyper_mat Hyperbolic functions
      sinh, cosh, tanh, etc.

      @defgroup complex_mat Complex operations
      real, imag, conjugate etc.
   @}

   @defgroup vector_mat Vector Algorithms
   @{

      sum, min, max, sort, set operations, etc.

      @defgroup reduce_mat Reduction operations
      sum, min, max, etc.

      @defgroup sort_mat Sort operations
      sort, sort by key, etc.

      @defgroup scan_mat Inclusive scan operations
      inclusive / cumulative sum, etc.

      @defgroup set_mat Set operations
      unique, union, intersect

      @defgroup calc_mat Numerical differentiation
      diff, gradient, etc.
   @}

   @defgroup linalg_mat Linear Algebra
   @{

     Matrix multiply, solve, decompositions

     @defgroup blas_mat BLAS operations
     Matrix multiply, dot product, etc.

     @defgroup lapack_factor_mat Matrix factorizations and decompositions
     LU, QR, Cholesky etc.

     @defgroup lapack_solve_mat Linear solve and least squares
     solve, solveLU, etc.

     @defgroup lapack_ops_mat Matrix operations
     inverse, det, rank, norm etc.
   @}

   @defgroup image_mat Image Processing
   @{

     Image filtering, morphing and transformations

     @defgroup colorconv_mat Colorspace conversions
     RGB to gray, gray to RGB, RGB to HSV, etc.

     @defgroup hist_mat Histograms
     Image and data histograms

     @defgroup transform_mat Image transformations
     rotate, skew, etc.

     @defgroup morph_mat Morphological Operations
     erode, dilate, etc.

     @defgroup imageflt_mat Filters
     bilateral, sobel, mean shift, median / min / max filters etc.

     @defgroup connected_comps_mat Connected Components & Labeling
     regions

     @defgroup utility_mat Utility Functions
     loadImage, saveImage, gaussianKernel
   @}

   @defgroup cv_mat Computer Vision
   @{

     A list of computer vision algorithms

     @defgroup featdetect_mat Feature detectors
     FAST feature detector

     @defgroup featdescriptor_mat Feature descriptors
     ORB feature descriptor

     @defgroup match_mat Template matching
   @}

   @defgroup signal_mat Signal Processing
   @{

     Convolutions, FFTs, filters

     @defgroup convolve_mat Convolutions
     1D, 2D and 3D convolutions

     @defgroup sigfilt_mat Filter
     fir, iir, etc.

     @defgroup fft_mat      Fast Fourier Transforms
     1D, 2D and 3D forward, inverse FFTs

     @defgroup approx_mat   Interpolation and approximation
     1D and 2D interpolation
   @}

   @defgroup stats_mat Statistics
   @{

     A list of Statistics functions
     @defgroup basicstats_mat Basic statistics functions
     mean, median, variance, etc.
   @}

   @defgroup io_mat Input and Output functions
   @{

     Functions to read and write data

     @defgroup dataio_mat Reading and writing arrays
     printing data to screen / files

     @defgroup imageio_mat Reading and writing images
     Reading and writing images
   @}

   @defgroup external Interface Functions
   @{

     CUDA/OpenCL specific functions

     @defgroup opencl_mat OpenCL specific functions

        \brief Accessing ArrayFire's context, queue, and share data with other OpenCL code.

        If your software is using ArrayFire's OpenCL backend, you can also write custom
        kernels and do custom memory operations using native OpenCL commands. The functions
        contained in the \p afcl namespace provide methods to get the context, queue, and
        device(s) that ArrayFire is using as well as convert `cl_mem` handles to
        \ref af::array objects.

        Please note: the \ref af::array constructors are not thread safe. You may create and
        upload data to `cl_mem` objects from separate threads, but the thread which
        instantiated ArrayFire must do the `cl_mem` to \ref af::array conversion.

   @}
@}


*/

/**
\example helloworld.cpp
\example pi.cpp
\example integer.cpp
\example rainfall.cpp
\example vectorize.cpp
\example black_scholes_options.cpp
\example monte_carlo_options.cpp
\example harris.cpp
\example kmeans.cpp
\example knn.cpp
\example bagging.cpp
\example naive_bayes.cpp
\example perceptron.cpp
\example neural_network.cpp
\example rbm.cpp
\example deep_belief_net.cpp
\example logistic_regression.cpp
\example conway.cpp
\example conway_pretty.cpp
\example fractal.cpp
\example histogram.cpp
\example plot2d.cpp
\example brain_segmentation.cpp
\example image_demo.cpp
\example morphing.cpp
\example optical_flow.cpp
\example pyramids.cpp
\example edge.cpp
*/

#include "af/compatible.h"
#include "af/algorithm.h"
#include "af/arith.h"
#include "af/array.h"
#include "af/blas.h"
#include "af/constants.h"
#include "af/complex.h"
#include "af/data.h"
#include "af/device.h"
#include "af/exception.h"
#include "af/features.h"
#include "af/gfor.h"
#include "af/graphics.h"
#include "af/image.h"
#include "af/index.h"
#include "af/lapack.h"
#include "af/seq.h"
#include "af/signal.h"
#include "af/statistics.h"
#include "af/timing.h"
#include "af/util.h"
#include "af/version.h"
#include "af/vision.h"
