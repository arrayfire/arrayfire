/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <inverse.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <err_cpu.hpp>
#include <handle.hpp>
#include <range.hpp>
#include <af/dim4.hpp>
#include <cassert>

#include <identity.hpp>
#include <lapack_helper.hpp>
#include <lu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <solve.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
using getri_func_def = int (*)(ORDER_TYPE, int, T *, int, const int *);

#define INV_FUNC_DEF(FUNC) \
    template<typename T>   \
    FUNC##_func_def<T> FUNC##_func();

#define INV_FUNC(FUNC, TYPE, PREFIX)            \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

INV_FUNC_DEF(getri)
INV_FUNC(getri, float, s)
INV_FUNC(getri, double, d)
INV_FUNC(getri, cfloat, c)
INV_FUNC(getri, cdouble, z)

template<typename T>
Array<T> inverse(const Array<T> &in) {
    int M = in.dims()[0];
    int N = in.dims()[1];

    if (M != N) {
        Array<T> I = identity<T>(in.dims());
        return solve(in, I);
    }

    Array<T> A       = copyArray<T>(in);
    Array<int> pivot = lu_inplace<T>(A, false);

    auto func = [=](Param<T> A, Param<int> pivot, int M) {
        getri_func<T>()(AF_LAPACK_COL_MAJOR, M, A.get(), A.strides(1),
                        pivot.get());
    };
    getQueue().enqueue(func, A, pivot, M);

    return A;
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace cpu
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace cpu
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
