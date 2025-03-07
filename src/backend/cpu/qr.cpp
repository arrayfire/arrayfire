/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <qr.hpp>

#include <err_cpu.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <lapack_helper.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <triangle.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace cpu {

template<typename T>
using geqrf_func_def = int (*)(ORDER_TYPE, int, int, T *, int, T *);

template<typename T>
using gqr_func_def = int (*)(ORDER_TYPE, int, int, int, T *, int, const T *);

#define QR_FUNC_DEF(FUNC) \
    template<typename T>  \
    FUNC##_func_def<T> FUNC##_func();

#define QR_FUNC(FUNC, TYPE, PREFIX)             \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

QR_FUNC_DEF(geqrf)
QR_FUNC(geqrf, float, s)
QR_FUNC(geqrf, double, d)
QR_FUNC(geqrf, cfloat, c)
QR_FUNC(geqrf, cdouble, z)

#define GQR_FUNC_DEF(FUNC) \
    template<typename T>   \
    FUNC##_func_def<T> FUNC##_func();

#define GQR_FUNC(FUNC, TYPE, PREFIX)            \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX);            \
    }

GQR_FUNC_DEF(gqr)
GQR_FUNC(gqr, float, sorgqr)
GQR_FUNC(gqr, double, dorgqr)
GQR_FUNC(gqr, cfloat, cungqr)
GQR_FUNC(gqr, cdouble, zungqr)

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    const dim4 NullShape(0, 0, 0, 0);

    dim4 endPadding(M - iDims[0], max(M, N) - iDims[1], 0, 0);
    q = (endPadding == NullShape
             ? copyArray(in)
             : padArrayBorders(in, NullShape, endPadding, AF_PAD_ZERO));
    q.resetDims(iDims);
    t = qr_inplace(q);

    // SPLIT into q and r
    dim4 rdims(M, N);
    r = createEmptyArray<T>(rdims);

    triangle<T>(r, q, true, false);

    auto func = [=](Param<T> q, Param<T> t, int M, int N) {
        gqr_func<T>()(AF_LAPACK_COL_MAJOR, M, M, min(M, N), q.get(),
                      q.strides(1), t.get());
    };
    q.resetDims(dim4(M, M));
    getQueue().enqueue(func, q, t, M, N);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];
    Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));

    auto func = [=](Param<T> in, Param<T> t, int M, int N) {
        geqrf_func<T>()(AF_LAPACK_COL_MAJOR, M, N, in.get(), in.strides(1),
                        t.get());
    };
    getQueue().enqueue(func, in, t, M, N);

    return t;
}

}  // namespace cpu
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace cpu {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

}  // namespace cpu
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace cpu {

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace cpu
}  // namespace arrayfire
