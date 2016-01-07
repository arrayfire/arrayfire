/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/blas.h>
#include <af/lapack.h>
#include <Array.hpp>
#include <memory.hpp>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif
#endif

// TODO: Ask upstream for a more official way to detect it
#ifdef OPENBLAS_CONST
#define IS_OPENBLAS
#endif

// Make sure we get the correct type signature for OpenBLAS
// OpenBLAS defines blasint as it's index type. Emulate this
// if we're not dealing with openblas and use it where applicable
#ifndef IS_OPENBLAS
typedef int blasint;
#endif

namespace opencl
{
namespace cpu
{
}
}

