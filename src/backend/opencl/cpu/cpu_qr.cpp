/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cpu/cpu_lapack_helper.hpp>
#include <cpu/cpu_qr.hpp>
#include <err_common.hpp>
#include <copy.hpp>

#include <af/dim4.hpp>
#include <handle.hpp>
#include <iostream>
#include <cassert>

#include <cpu/cpu_triangle.hpp>

namespace opencl
{
namespace cpu
{

template<typename T>
using geqrf_func_def = int (*)(ORDER_TYPE, int, int,
                               T*, int,
                               T*);

template<typename T>
using gqr_func_def = int (*)(ORDER_TYPE, int, int, int,
                             T*, int,
                             const T*);

#define QR_FUNC_DEF( FUNC )                                         \
template<typename T> FUNC##_func_def<T> FUNC##_func();


#define QR_FUNC( FUNC, TYPE, PREFIX )                               \
template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()            \
{ return & LAPACK_NAME(PREFIX##FUNC); }

QR_FUNC_DEF( geqrf )
QR_FUNC(geqrf , float  , s)
QR_FUNC(geqrf , double , d)
QR_FUNC(geqrf , cfloat , c)
QR_FUNC(geqrf , cdouble, z)

#define GQR_FUNC_DEF( FUNC )                                         \
template<typename T> FUNC##_func_def<T> FUNC##_func();

#define GQR_FUNC( FUNC, TYPE, PREFIX )                               \
template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()             \
{ return & LAPACK_NAME(PREFIX); }

GQR_FUNC_DEF( gqr )
GQR_FUNC(gqr , float  , sorgqr)
GQR_FUNC(gqr , double , dorgqr)
GQR_FUNC(gqr , cfloat , cungqr)
GQR_FUNC(gqr , cdouble, zungqr)

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    dim4 padDims(M, max(M, N));
    q = padArray<T, T>(in, padDims, scalar<T>(0));
    q.resetDims(iDims);
    t = qr_inplace(q);

    // SPLIT into q and r
    dim4 rdims(M, N);
    r = createEmptyArray<T>(rdims);

    T *qPtr = getMappedPtr<T>(q.get());
    T *rPtr = getMappedPtr<T>(r.get());
    T *tPtr = getMappedPtr<T>(t.get());

    triangle<T, true, false>(rPtr, qPtr, rdims, r.strides(), q.strides());

    gqr_func<T>()(AF_LAPACK_COL_MAJOR,
                  M, M, min(M, N),
                  qPtr, q.strides()[1],
                  tPtr);

    unmapPtr(q.get(), qPtr);
    unmapPtr(r.get(), rPtr);
    unmapPtr(t.get(), tPtr);

    q.resetDims(dim4(M, M));
}

template<typename T>
Array<T> qr_inplace(Array<T> &in)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));

    T *iPtr = getMappedPtr<T>(in.get());
    T *tPtr = getMappedPtr<T>(t.get());

    geqrf_func<T>()(AF_LAPACK_COL_MAJOR, M, N,
                    iPtr, in.strides()[1],
                    tPtr);

    unmapPtr(in.get(), iPtr);
    unmapPtr(t.get(), tPtr);

    return t;
}

#define INSTANTIATE_QR(T)                                                                           \
    template Array<T> qr_inplace<T>(Array<T> &in);                                                \
    template void qr<T>(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}
}
