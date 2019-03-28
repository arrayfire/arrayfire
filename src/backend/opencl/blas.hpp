/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>

// This file contains the common interface for OpenCL BLAS
// functions. They can be implemented in different back-ends,
// such as CLBlast or clBLAS.

namespace opencl {

void initBlas();
void deInitBlas();

template<typename T>
void gemm(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs,
          const T *alpha, const Array<T> &lhs, const Array<T> &rhs,
          const T *beta);

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs);
}  // namespace opencl
