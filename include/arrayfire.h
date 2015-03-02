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

@defgroup cpp_interface C++ Interface
@{

   @defgroup array_mem Array Member Functions
   @{
    @defgroup array_construct Constructors
    @defgroup array_operator Operator Overloads
    @defgroup array_mem_func Member Functions
    @defgroup array_indexing Indexing
   @}

   @defgroup basics_mat Basic operations
   @{
     @defgroup construct Array allocation, initialization
     @defgroup gen Generate and fill Arrays
     @defgroup arith Element-wise arithmetic
     @defgroup indexing Subscripted array indexing
     @defgroup props Array info (dimensions, type, complexity, ...)
     @defgroup manip_mat Array manipulation (transpose, join, tile, shift, ...)
     @defgroup data Device memory management, access
     @defgroup gfor Parallelized loops: gfor
     @defgroup device_mat Device management
     @defgroup disp Displaying variables
   @}

   @defgroup data_mat Data Analysis
   @{
     @defgroup summul_mat Sum and Product values
     @defgroup minmax_mat Minimum and Maximum values
     @defgroup test_mat Test if any/all true
     @defgroup where_mat Index or count nonzero elements
     @defgroup accum_mat Cumulative and segmented Sum
     @defgroup stats_mat Statistics: average, median, variance, histogram, etc.
     @defgroup grad_mat Gradient or grid construction
     @defgroup set_mat Set operations: union, unique, intersection, ...
     @defgroup sort_mat Sorting (vectors, columns, rows)
     @defgroup hist_mat Histograms
   @}

   @defgroup linalg_mat Linear Algebra
   @{
     @defgroup blas_mat Matrix multiply, dot product, BLAS
     @defgroup linsolve_mat Solving linear systems
     @defgroup factor_mat Factorizations: LU, QR, Cholesky, singular values, eigenvalues, Hessenberg
     @defgroup matops_mat Matrix Operations
   @}

   @defgroup image_mat Image and Signal Processing
   @{
     @defgroup morph_mat Morphing: erosion, dilation, ...
     @defgroup transform_mat Image transformations
     @defgroup colorconv_mat Colorspace conversions
     @defgroup hist_mat Histograms
     @defgroup filter_mat Signal filtering
     @defgroup props_mat Connected components, labeling, centroids, ...
     @defgroup fft_mat Fourier transforms (1D, 2D, 3D)
     @defgroup convolution_mat Image filtering & convolutions
     @defgroup interpolation Interpolation and rescaling
     @defgroup image_util Utility Image Functions
   @}

@}

\defgroup c_interface C Interface
@{
    \defgroup arr_basic Basic Functions(Allocation, Copy, Destroy)
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
#include "af/graphics.h"
#include "af/image.h"
#include "af/index.h"
#include "af/reduce.h"
#include "af/seq.h"
#include "af/signal.h"
#include "af/statistics.h"
#include "af/timing.h"
#include "af/util.h"
