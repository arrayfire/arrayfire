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

   @defgroup array_mat Functions to create Arrays
   @{
      // array constructors
      // data generators
   @}

   @defgroup manip_mat Manipulating Arrays
   @{
      // Indexing
      // moddims, flat
      // reorder, transpose
      // join
      // flip
   @}

   @defgroup arith_mat Mathematical functions
   @{
     // arithmetic (+, -, /, *)
     // logical
     // numeric (round, ceil, floor, min, max etc)
     // trigonometric
     // exponential / logarithmic
     // complex
   @}

   @defgroup vector_mat Vector Algorithms
   @{
      @defgroup reduce_mat Reduction operations (Sum, Min, Max, etc)
      @defgroup sort_mat Sort operations
      @defgroup scan_mat Inclusive scan operations (accum, where, etc)
      @defgroup set_mat Set operations (unique, union, intersect)
      @defgroup calc_mat Numerical differentiation (diff, gradient)
   @}

   @defgroup linalg_mat Linear Algebra
   @{
     @defgroup blas_mat Matrix multiply, dot product, transpose
   @}

   @defgroup image_mat Image Processing
   @{
     @defgroup colorconv_mat Colorspace conversions
     @defgroup hist_mat Histograms
     // Image transformations
     // Morph
     // filters: sobel, bilateral, meanshift
   @}

   @defgroup cv_mat Computer Vision
   @{
     @defgroup featdetect_mat Feature detectors
     @defgroup featdescriptor_mat Feature descriptors
     // template matching
   @}

   @defgroup signal_mat Signal Processing
   @{
      // Convolve
      // FFT
      // approx
   @}

   @defgroup stats_mat Statistics
   @{
      // for now just one group with mean, var, std etc
   @}

   @defgroup io_mat Input and Output functions
   @{
   // af_print
   // loadImage and saveImage
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
