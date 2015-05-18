/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <lu.hpp>
#include <err_common.hpp>

#if defined(WITH_CPU_LINEAR_ALGEBRA)

#include <af/dim4.hpp>
#include <handle.hpp>
#include <iostream>
#include <cassert>
#include <err_cpu.hpp>

#include <range.hpp>
#include <lapack_helper.hpp>

namespace cpu
{

template<typename T>
using getrf_func_def = int (*)(ORDER_TYPE, int, int,
                               T*, int,
                               int*);

#define LU_FUNC_DEF( FUNC )                                     \
template<typename T> FUNC##_func_def<T> FUNC##_func();


#define LU_FUNC( FUNC, TYPE, PREFIX )                           \
template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()        \
{ return & LAPACK_NAME(PREFIX##FUNC); }

LU_FUNC_DEF( getrf )
LU_FUNC(getrf , float  , s)
LU_FUNC(getrf , double , d)
LU_FUNC(getrf , cfloat , c)
LU_FUNC(getrf , cdouble, z)

template<typename T>
void lu_split(Array<T> &lower, Array<T> &upper, const Array<T> &in)
{
    T *l = lower.get();
    T *u = upper.get();
    const T *i = in.get();

    dim4 ldm = lower.dims();
    dim4 udm = upper.dims();
    dim4 idm = in.dims();

    dim4 lst = lower.strides();
    dim4 ust = upper.strides();
    dim4 ist = in.strides();

    for(dim_t ow = 0; ow < idm[3]; ow++) {
        const dim_t lW = ow * lst[3];
        const dim_t uW = ow * ust[3];
        const dim_t iW = ow * ist[3];

        for(dim_t oz = 0; oz < idm[2]; oz++) {
            const dim_t lZW = lW + oz * lst[2];
            const dim_t uZW = uW + oz * ust[2];
            const dim_t iZW = iW + oz * ist[2];

            for(dim_t oy = 0; oy < idm[1]; oy++) {
                const dim_t lYZW = lZW + oy * lst[1];
                const dim_t uYZW = uZW + oy * ust[1];
                const dim_t iYZW = iZW + oy * ist[1];

                for(dim_t ox = 0; ox < idm[0]; ox++) {
                    const dim_t lMem = lYZW + ox;
                    const dim_t uMem = uYZW + ox;
                    const dim_t iMem = iYZW + ox;
                    if(ox > oy) {
                        if(oy < ldm[1])
                            l[lMem] = i[iMem];
                        if(ox < udm[0])
                            u[uMem] = scalar<T>(0);
                    } else if (oy > ox) {
                        if(oy < ldm[1])
                            l[lMem] = scalar<T>(0);
                        if(ox < udm[0])
                            u[uMem] = i[iMem];
                    } else if(ox == oy) {
                        if(oy < ldm[1])
                            l[lMem] = scalar<T>(1.0);
                        if(ox < udm[0])
                            u[uMem] = i[iMem];
                    }
                }
            }
        }
    }
}

void convertPivot(Array<int> &pivot, int out_sz)
{
    Array<int> p = range<int>(dim4(out_sz), 0);
    int *d_pi = pivot.get();
    int *d_po = p.get();
    dim_t d0 = pivot.dims()[0];
    for(int j = 0; j < (int)d0; j++) {
        // 1 indexed in pivot
        std::swap(d_po[j], d_po[d_pi[j] - 1]);
    }
    pivot = p;
}

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    Array<T> in_copy = copyArray<T>(in);
    pivot = lu_inplace(in_copy);

    // SPLIT into lower and upper
    dim4 ldims(M, min(M, N));
    dim4 udims(min(M, N), N);
    lower = createEmptyArray<T>(ldims);
    upper = createEmptyArray<T>(udims);

    lu_split<T>(lower, upper, in_copy);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    Array<int> pivot = createEmptyArray<int>(af::dim4(min(M, N), 1, 1, 1));

    getrf_func<T>()(AF_LAPACK_COL_MAJOR, M, N,
                    in.get(), in.strides()[1],
                    pivot.get());

    if(convert_pivot) convertPivot(pivot, M);

    return pivot;
}

#define INSTANTIATE_LU(T)                                                                           \
    template Array<int> lu_inplace<T>(Array<T> &in, const bool convert_pivot);                      \
    template void lu<T>(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}

#else

namespace cpu
{

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot)
{
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_LU(T)                                                                           \
    template Array<int> lu_inplace<T>(Array<T> &in, const bool convert_pivot);                      \
    template void lu<T>(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}

#endif
