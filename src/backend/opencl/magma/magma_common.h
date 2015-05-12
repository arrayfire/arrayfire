/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __MAGMA_COMMON_H
#define __MAGMA_COMMON_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define HAVE_clBLAS
#include <clBLAS.h>

#include "magma_types.h"

#define magma_s magmaFloat_ptr
#define magma_d magmaDouble_ptr
#define magma_c magmaFloatComplex_ptr
#define magma_z magmaDoubleComplex_ptr

#define magmablas_s magmaFloat_ptr
#define magmablas_d magmaDouble_ptr
#define magmablas_c magmaFloatComplex_ptr
#define magmablas_z magmaDoubleComplex_ptr

#endif
