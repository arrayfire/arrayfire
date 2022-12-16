/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cholesky.hpp>

#include <common/err_common.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <Array.hpp>
#include <Param.hpp>
#include <copy.hpp>
#include <types.hpp>

#include <lapack_helper.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <triangle.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
using potrf_func_def = int (*)(ORDER_TYPE, char, int, T *, int);

#define CH_FUNC_DEF(FUNC) \
    template<typename T>  \
    FUNC##_func_def<T> FUNC##_func();

#define CH_FUNC(FUNC, TYPE, PREFIX)             \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

CH_FUNC_DEF(potrf)
CH_FUNC(potrf, float, s)
CH_FUNC(potrf, double, d)
CH_FUNC(potrf, cfloat, c)
CH_FUNC(potrf, cdouble, z)

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    Array<T> out = copyArray<T>(in);
    *info        = cholesky_inplace(out, is_upper);

    triangle<T>(out, out, is_upper, false);

    return out;
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    dim4 iDims = in.dims();
    int N      = iDims[0];

    char uplo = 'L';
    if (is_upper) { uplo = 'U'; }

    int info  = 0;
    auto func = [&](int *info, Param<T> in) {
        *info = potrf_func<T>()(AF_LAPACK_COL_MAJOR, uplo, N, in.get(),
                                in.strides(1));
    };

    getQueue().enqueue(func, &info, in);
    // Ensure the value of info has been written into info.
    getQueue().sync();

    return info;
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace cpu
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace cpu
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
