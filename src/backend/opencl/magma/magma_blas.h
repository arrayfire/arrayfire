/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __MAGMA_BLAS_H
#define __MAGMA_BLAS_H

// This file contains the common interface for Magma OpenCL BLAS
// functions. They can be implemented in different back-ends,
// such as CLBlast or clBLAS.

#include <types.hpp>
#include "magma_common.h"

using opencl::cdouble;
using opencl::cfloat;

template<typename T>
struct gpu_blas_gemm_func;
template<typename T>
struct gpu_blas_gemv_func;
template<typename T>
struct gpu_blas_trmm_func;
template<typename T>
struct gpu_blas_trsm_func;
template<typename T>
struct gpu_blas_trsv_func;
template<typename T>
struct gpu_blas_herk_func;

#if defined(USE_CLBLAST)
#include "magma_blas_clblast.h"
#endif

#if defined(USE_CLBLAS)
#include "magma_blas_clblas.h"
#endif

#endif  // __MAGMA_BLAS_H
