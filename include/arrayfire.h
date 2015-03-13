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

\defgroup arrayfire_func ArrayFire Functions
@{
@}

@defgroup func_categories Function Categories
@{

   @defgroup array_mat Functions to create and modify Arrays
   @{
      @defgroup array_construct Constructors of array class

      @defgroup data_mat Functions to create arrays.
      constant, random, range, etc

      @defgroup index_mat Indexing operation on arrays

      @defgroup order_mat Reorder array content
      reorder, transpose, flip, etc

      @defgroup move_mat Joining and tiling operations
      join, tile, etc

      @defgroup array_basic C functions to create af_array
   @}

   @defgroup mathfunc_mat Mathematical functions
   @{

      @defgroup arith_mat Arithmetic operations
      +, -, *, /, >>, <<

      @defgroup logic_mat  Logical operations
      &&, ||, |, &, <, >, <=, >=, ==, !

      @defgroup numeric_mat Numeric functions
      floor, round, min, max, etc

      @defgroup trig_mat Trigonometric functions
      sin, cos, tan, etc

      @defgroup explog_mat Exponential and logarithmic functions
      exp, log, expm1, log1p, etc

      @defgroup hyper_mat Hyperbolic functions
      sinh, cosh, tanh, etc

      @defgroup complex_mat Complex operations
      real, imag, conjugate etc
   @}

   @defgroup vector_mat Vector Algorithms
   @{
      @defgroup reduce_mat Reduction operations
      sum, min, max, etc

      @defgroup sort_mat Sort operations
      sort, sort by key, etc

      @defgroup scan_mat Inclusive scan operations
      inclusive / cumulative sum, etc

      @defgroup set_mat Set operations
      unique, union, intersect

      @defgroup calc_mat Numerical differentiation
      diff, gradient, etc
   @}

   @defgroup linalg_mat Linear Algebra
   @{
     @defgroup blas_mat BLAS operations
     Matrix multiply, dot product, etc
   @}

   @defgroup image_mat Image Processing
   @{
     @defgroup colorconv_mat Colorspace conversions
     RGB to gray, gray to RGB, RGB to HSV, etc

     @defgroup hist_mat Histograms
     Image and data histograms

     @defgroup transform_mat Image transformations
     rotate, skew, etc

     @defgroup morph_mat Image morphing operations
     erode, dilate, etc

     @defgroup imageflt_mat Image filtering operators
     bilateral, sobel, mean shift, etc

     @defgroup connected_comps_mat Connected Components & Labeling
     regions
   @}

   @defgroup cv_mat Computer Vision
   @{
     @defgroup featdetect_mat Feature detectors
     FAST feature detector

     @defgroup featdescriptor_mat Feature descriptors
     ORB feature descriptor

     @defgroup match_mat Template matching
   @}

   @defgroup signal_mat Signal Processing
   @{

     @defgroup convolve_mat Convolutions
     1D, 2D and 3D convolutions

     @defgroup fft_mat      Fast Fourier Transforms
     1D, 2D and 3D forward, inverse FFTs

     @defgroup approx_mat   Interpolation and approximation
   @}

   @defgroup stats_mat Statistics
   @{
     @defgroup basicstats_mat Basic statistics functions
     mean, median, variance, etc
   @}

   @defgroup io_mat Input and Output functions
   @{
     @defgroup dataio_mat Reading and writing arrays
     printing data to screen / files

     @defgroup imageio_mat Reading and writing images
     Reading and writing images
   @}
@}

*/

#include "af/compatible.h"
#include "af/algorithm.h"
#include "af/arith.h"
#include "af/array.h"
#include "af/blas.h"
#include "af/data.h"
#include "af/device.h"
#include "af/exception.h"
#include "af/features.h"
#include "af/gfor.h"
#include "af/image.h"
#include "af/index.h"
#include "af/seq.h"
#include "af/signal.h"
#include "af/statistics.h"
#include "af/timing.h"
#include "af/util.h"
