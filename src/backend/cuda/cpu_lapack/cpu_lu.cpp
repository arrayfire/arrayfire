/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_CPU_LINEAR_ALGEBRA)

#include <cpu_lapack/cpu_lu.hpp>
#include <err_common.hpp>

#include <af/dim4.hpp>
#include <handle.hpp>
#include <iostream>
#include <cassert>
#include <err_cuda.hpp>

#include "lapack_helper.hpp"

namespace cuda
{
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
void lu_split(T *l, T *u, const T *i,
        const dim4 ldm, const dim4 udm, const dim4 idm,
        const dim4 lst, const dim4 ust, const dim4 ist)
{
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

void convertPivot(int **pivot, int out_sz, dim_t d0)
{
    int* p = pinnedAlloc<int>(out_sz);
    for(int i = 0; i < out_sz; i++)
        p[i] = i;

    for(int j = 0; j < (int)d0; j++) {
        // 1 indexed in pivot
        std::swap(p[j], p[(*pivot)[j] - 1]);
    }

    pinnedFree(*pivot);
    *pivot = p;
}

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    Array<T> in_copy = copyArray<T>(in);

    //////////////////////////////////////////
    // LU inplace
    int *pivotPtr  = pinnedAlloc<int>(min(M, N));
    T   *inPtr     = pinnedAlloc<T>  (in_copy.elements());
    copyData(inPtr, in);

    getrf_func<T>()(AF_LAPACK_COL_MAJOR, M, N,
                    inPtr, in_copy.strides()[1],
                    pivotPtr);

    convertPivot(&pivotPtr, M, min(M, N));

    pivot = createHostDataArray<int>(af::dim4(M), pivotPtr);
    //////////////////////////////////////////

    // SPLIT into lower and upper
    dim4 ldims(M, min(M, N));
    dim4 udims(min(M, N), N);

    T *lowerPtr = pinnedAlloc<T>(ldims.elements());
    T *upperPtr = pinnedAlloc<T>(udims.elements());

    dim4 lst(1, ldims[0], ldims[0] * ldims[1], ldims[0] * ldims[1] * ldims[2]);
    dim4 ust(1, udims[0], udims[0] * udims[1], udims[0] * udims[1] * udims[2]);

    lu_split<T>(lowerPtr, upperPtr, inPtr, ldims, udims, iDims,
                lst, ust, in_copy.strides());

    lower = createHostDataArray<T>(ldims, lowerPtr);
    upper = createHostDataArray<T>(udims, upperPtr);

    lower.eval();
    upper.eval();

    pinnedFree(lowerPtr);
    pinnedFree(upperPtr);
    pinnedFree(pivotPtr);
    pinnedFree(inPtr);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    int *pivotPtr  = pinnedAlloc<int>(min(M, N));
    T   *inPtr     = pinnedAlloc<T>  (in.elements());
    copyData(inPtr, in);

    getrf_func<T>()(AF_LAPACK_COL_MAJOR, M, N,
                    inPtr, in.strides()[1],
                    pivotPtr);

    if(convert_pivot) convertPivot(&pivotPtr, M, min(M, N));

    writeHostDataArray<T>(in, inPtr, in.elements() * sizeof(T));
    Array<int> pivot = createHostDataArray<int>(af::dim4(M), pivotPtr);

    pivot.eval();

    pinnedFree(inPtr);
    pinnedFree(pivotPtr);

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
}

#endif
