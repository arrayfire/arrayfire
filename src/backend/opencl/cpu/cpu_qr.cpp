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
#include <cpu/cpu_helper.hpp>
#include <cpu/cpu_qr.hpp>
#include <cpu/cpu_triangle.hpp>

namespace arrayfire {
namespace opencl {
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

    mapped_ptr<T> qPtr = q.getMappedPtr();
    mapped_ptr<T> rPtr = r.getMappedPtr();
    mapped_ptr<T> tPtr = t.getMappedPtr();

    triangle<T, true, false>(rPtr.get(), qPtr.get(), rdims, r.strides(),
                             q.strides());

    gqr_func<T>()(AF_LAPACK_COL_MAJOR, M, M, min(M, N), qPtr.get(),
                  q.strides()[1], tPtr.get());

    q.resetDims(dim4(M, M));
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));

    mapped_ptr<T> iPtr = in.getMappedPtr();
    mapped_ptr<T> tPtr = t.getMappedPtr();

    geqrf_func<T>()(AF_LAPACK_COL_MAJOR, M, N, iPtr.get(), in.strides()[1],
                    tPtr.get());

    return t;
}

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace cpu
}  // namespace opencl
}  // namespace arrayfire
#endif  // WITH_LINEAR_ALGEBRA
