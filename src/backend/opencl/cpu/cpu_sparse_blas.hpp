/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/SparseArray.hpp>

#ifdef USE_MKL
#include <mkl_spblas.h>
#endif

#ifdef USE_MKL
using sp_cfloat  = MKL_Complex8;
using sp_cdouble = MKL_Complex16;
#else
using sp_cfloat  = arrayfire::opencl::cfloat;
using sp_cdouble = arrayfire::opencl::cdouble;
#endif

namespace arrayfire {
namespace opencl {
namespace cpu {

template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs);

}  // namespace cpu
}  // namespace opencl
}  // namespace arrayfire
