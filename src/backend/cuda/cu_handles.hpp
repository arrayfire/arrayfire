/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

namespace cuda {
using BlasHandle = cublasHandle_t;
using SolveHandle = cusolverDnHandle_t;
using SparseHandle = cusparseHandle_t ;
}
