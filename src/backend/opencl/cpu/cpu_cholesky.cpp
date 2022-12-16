/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <cpu/cpu_cholesky.hpp>
#include <cpu/cpu_helper.hpp>
#include <cpu/cpu_triangle.hpp>

namespace arrayfire {
namespace opencl {
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

    mapped_ptr<T> oPtr = out.getMappedPtr();

    if (is_upper) {
        triangle<T, true, false>(oPtr.get(), oPtr.get(), out.dims(),
                                 out.strides(), out.strides());
    } else {
        triangle<T, false, false>(oPtr.get(), oPtr.get(), out.dims(),
                                  out.strides(), out.strides());
    }

    return out;
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    dim4 iDims = in.dims();
    int N      = iDims[0];

    char uplo = 'L';
    if (is_upper) { uplo = 'U'; }

    mapped_ptr<T> inPtr = in.getMappedPtr();

    int info = potrf_func<T>()(AF_LAPACK_COL_MAJOR, uplo, N, inPtr.get(),
                               in.strides()[1]);

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
}  // namespace opencl
}  // namespace arrayfire
#endif  // WITH_LINEAR_ALGEBRA
