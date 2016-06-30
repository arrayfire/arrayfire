/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>

#include <stdexcept>
#include <string>
#include <cassert>

#include <af/dim4.hpp>
#include <complex.hpp>
#include <handle.hpp>
#include <err_common.hpp>
#include <math.hpp>
#include <platform.hpp>

namespace opencl
{

using namespace common;

template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    Array<T> out = createValueArray<T>(af::dim4(lhs.dims()[0], rhs.dims()[0], 1, 1), scalar<T>(0));
    return out;
}

#define INSTANTIATE_SPARSE(T)                                                           \
    template Array<T> matmul<T>(const common::SparseArray<T> lhs, const Array<T> rhs,   \
                                af_mat_prop optLhs, af_mat_prop optRhs);                \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}
