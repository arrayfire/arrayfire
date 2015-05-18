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

#if defined(WITH_CUDA_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>
#include <memory.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <err_common.hpp>

#include <kernel/lu_split.hpp>

namespace cuda
{

using cusolver::getDnHandle;

//cusolverStatus_t CUDENSEAPI cusolverDn<>getrf_bufferSize(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A,
//        int lda, int *Lwork );
//
//
//cusolverStatus_t CUDENSEAPI cusolverDn<>getrf(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A,
//        int lda,
//        <> *Workspace,
//        int *devIpiv, int *devInfo );

template<typename T>
struct getrf_func_def_t
{
    typedef cusolverStatus_t (*getrf_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int,
                              T *,
                              int *, int *);
};

template<typename T>
struct getrf_buf_func_def_t
{
    typedef cusolverStatus_t (*getrf_buf_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int, int *);
};

#define LU_FUNC_DEF( FUNC )                                                     \
template<typename T>                                                            \
typename FUNC##_func_def_t<T>::FUNC##_func_def                                  \
FUNC##_func();                                                                  \
                                                                                \
template<typename T>                                                            \
typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def                          \
FUNC##_buf_func();


#define LU_FUNC( FUNC, TYPE, PREFIX )                                                   \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def                            \
FUNC##_func<TYPE>()                                                                     \
{ return (FUNC##_func_def_t<TYPE>::FUNC##_func_def)&cusolverDn##PREFIX##FUNC; }         \
                                                                                        \
template<> typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def                    \
FUNC##_buf_func<TYPE>()                                                                 \
{ return (FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def)& cusolverDn##PREFIX##FUNC##_bufferSize; }

LU_FUNC_DEF( getrf )
LU_FUNC(getrf , float  , S)
LU_FUNC(getrf , double , D)
LU_FUNC(getrf , cfloat , C)
LU_FUNC(getrf , cdouble, Z)

void convertPivot(Array<int> &pivot, int out_sz)
{
    dim_t d0 = pivot.dims()[0];

    std::vector<int> d_po(out_sz);
    for(int i = 0; i < out_sz; i++) {
        d_po[i] = i;
    }

    std::vector<int> d_pi(d0);
    copyData(&d_pi[0], pivot);

    for(int j = 0; j < d0; j++) {
        // 1 indexed in pivot
        std::swap(d_po[j], d_po[d_pi[j] - 1]);
    }

    pivot = createHostDataArray<int>(out_sz, &d_po[0]);
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
    kernel::lu_split<T>(lower, upper, in_copy);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    Array<int> pivot = createEmptyArray<int>(af::dim4(min(M, N), 1, 1, 1));

    int lwork = 0;

    CUSOLVER_CHECK(getrf_buf_func<T>()(getDnHandle(),
                                       M, N,
                                       in.get(), in.strides()[1],
                                       &lwork));

    T *workspace = memAlloc<T>(lwork);
    int *info = memAlloc<int>(1);

    CUSOLVER_CHECK(getrf_func<T>()(getDnHandle(),
                                   M, N,
                                   in.get(), in.strides()[1],
                                   workspace,
                                   pivot.get(),
                                   info));

    if(convert_pivot) convertPivot(pivot, M);

    memFree(workspace);
    memFree(info);

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
namespace cuda
{
template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
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
