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
#include <range.hpp>
#include <lapack_helper.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/lu.hpp>

namespace cpu
{

template<typename T>
using getrf_func_def = int (*)(ORDER_TYPE, int, int, T*, int, int*);

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
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    lower.eval();
    upper.eval();
    pivot.eval();
    in.eval();

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

    getQueue().enqueue(kernel::lu_split<T>, lower, upper, in_copy);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot)
{
    in.eval();

    dim4 iDims = in.dims();
    Array<int> pivot = createEmptyArray<int>(af::dim4(min(iDims[0], iDims[1]), 1, 1, 1));

    auto func = [=] (Param<T> in, Param<int> pivot) {
        dim4 iDims = in.dims;
        getrf_func<T>()(AF_LAPACK_COL_MAJOR, iDims[0], iDims[1], in.get(), in.strides[1], pivot.get());
    };
    getQueue().enqueue(func, in, pivot);

    if(convert_pivot) {
        Array<int> p = range<int>(dim4(iDims[0]), 0);
        getQueue().enqueue(kernel::convertPivot, p, pivot);
        return p;
    } else {
        return pivot;
    }
}

bool isLAPACKAvailable()
{
    return true;
}

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

bool isLAPACKAvailable()
{
    return false;
}

}

#endif

namespace cpu
{

#define INSTANTIATE_LU(T)                                                                           \
    template Array<int> lu_inplace<T>(Array<T> &in, const bool convert_pivot);                      \
    template void lu<T>(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}
