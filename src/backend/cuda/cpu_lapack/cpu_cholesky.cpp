/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cholesky.hpp>
#include <err_common.hpp>

#if defined(WITH_CPU_LINEAR_ALGEBRA)

#include <af/dim4.hpp>
#include <handle.hpp>
#include <copy.hpp>
#include <iostream>
#include <cassert>

#include <cpu_lapack/cpu_triangle.hpp>
#include "lapack_helper.hpp"

namespace cuda
{
namespace cpu
{

template<typename T>
using potrf_func_def = int (*)(ORDER_TYPE, char,
                               int,
                               T*, int);

#define CH_FUNC_DEF( FUNC )                                     \
template<typename T> FUNC##_func_def<T> FUNC##_func();


#define CH_FUNC( FUNC, TYPE, PREFIX )                           \
template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()        \
{ return & LAPACK_NAME(PREFIX##FUNC); }

CH_FUNC_DEF( potrf )
CH_FUNC(potrf , float  , s)
CH_FUNC(potrf , double , d)
CH_FUNC(potrf , cfloat , c)
CH_FUNC(potrf , cdouble, z)

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper)
{
    dim4 iDims = in.dims();
    int N = iDims[0];

    char uplo = 'L';
    if(is_upper)
        uplo = 'U';

    T *inPtr = pinnedAlloc<T>(in.elements());
    copyData(inPtr, in);

    *info = potrf_func<T>()(AF_LAPACK_COL_MAJOR, uplo,
                            N, inPtr, in.strides()[1]);

    if (is_upper) triangle<T, true , false>(inPtr, inPtr, in.dims(), in.strides(), in.strides());
    else          triangle<T, false, false>(inPtr, inPtr, in.dims(), in.strides(), in.strides());

    Array<T> out = createHostDataArray<T>(in.dims(), inPtr);

    pinnedFree(inPtr);

    return out;
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper)
{
    dim4 iDims = in.dims();
    int N = iDims[0];

    char uplo = 'L';
    if(is_upper)
        uplo = 'U';

    T *inPtr = pinnedAlloc<T>(in.elements());
    copyData(inPtr, in);

    int info = potrf_func<T>()(AF_LAPACK_COL_MAJOR, uplo,
                               N, inPtr, in.strides()[1]);

    writeHostDataArray<T>(in, inPtr, in.elements() * sizeof(T));

    pinnedFree(inPtr);

    return info;
}

#define INSTANTIATE_CH(T)                                                                   \
    template int cholesky_inplace<T>(Array<T> &in, const bool is_upper);                    \
    template Array<T> cholesky<T>   (int *info, const Array<T> &in, const bool is_upper);   \


INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}
}

#endif
