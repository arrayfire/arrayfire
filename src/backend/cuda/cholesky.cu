/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <cholesky.hpp>

#if defined(WITH_CUDA_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>
#include <cublas_v2.h>
#include <identity.hpp>
#include <iostream>
#include <memory.hpp>
#include <copy.hpp>
#include <triangle.hpp>

#include <math.hpp>
#include <err_common.hpp>

namespace cuda
{

using cusolver::getDnHandle;

//cusolverStatus_t cusolverDn<>potrf_bufferSize(
//        cusolverDnHandle_t handle,
//        cublasFillMode_t uplo,
//        int n,
//        <> *A,
//        int lda,
//        int *Lwork );
//
//cusolverStatus_t cusolverDn<>potrf(
//        cusolverDnHandle_t handle,
//        cublasFillMode_t uplo,
//        int n,
//        <> *A, int lda,
//        <> *Workspace, int Lwork,
//        int *devInfo );

template<typename T>
struct potrf_func_def_t
{
    typedef cusolverStatus_t (*potrf_func_def) (
                              cusolverDnHandle_t,
                              cublasFillMode_t,
                              int,
                              T *, int,
                              T *,
                              int, int *);
};

template<typename T>
struct potrf_buf_func_def_t
{
    typedef cusolverStatus_t (*potrf_buf_func_def) (
                              cusolverDnHandle_t,
                              cublasFillMode_t,
                              int,
                              T *, int,
                              int *);
};

#define CH_FUNC_DEF( FUNC )                                                     \
template<typename T>                                                            \
typename FUNC##_func_def_t<T>::FUNC##_func_def                                  \
FUNC##_func();                                                                  \
                                                                                \
template<typename T>                                                            \
typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def                          \
FUNC##_buf_func();                                                              \

#define CH_FUNC( FUNC, TYPE, PREFIX )                                                           \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>()                \
{ return (FUNC##_func_def_t<TYPE>::FUNC##_func_def)&cusolverDn##PREFIX##FUNC; }                 \
                                                                                                \
template<> typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def FUNC##_buf_func<TYPE>()    \
{ return (FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def)&cusolverDn##PREFIX##FUNC##_bufferSize; }

CH_FUNC_DEF( potrf )
CH_FUNC(potrf , float  , S)
CH_FUNC(potrf , double , D)
CH_FUNC(potrf , cfloat , C)
CH_FUNC(potrf , cdouble, Z)

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper)
{

    Array<T> out = copyArray<T>(in);
    *info = cholesky_inplace(out, is_upper);

    if (is_upper) triangle<T, true , false>(out, out);
    else          triangle<T, false, false>(out, out);

    return out;
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper)
{
    dim4 iDims = in.dims();
    int N = iDims[0];

    int lwork = 0;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    if(is_upper)
        uplo = CUBLAS_FILL_MODE_UPPER;

    CUSOLVER_CHECK(potrf_buf_func<T>()(getDnHandle(),
                                       uplo,
                                       N,
                                       in.get(), in.strides()[1],
                                       &lwork));

    T *workspace = memAlloc<T>(lwork);
    int *d_info = memAlloc<int>(1);

    CUSOLVER_CHECK(potrf_func<T>()(getDnHandle(),
                                   uplo,
                                   N,
                                   in.get(), in.strides()[1],
                                   workspace, lwork,
                                   d_info));

    memFree(workspace);
    memFree(d_info);

    //FIXME: should return h_info
    return 0;
}

#define INSTANTIATE_CH(T)                                                                   \
    template int cholesky_inplace<T>(Array<T> &in, const bool is_upper);                    \
    template Array<T> cholesky<T>   (int *info, const Array<T> &in, const bool is_upper);   \


INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)
}

#else
namespace cuda
{

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_CH(T)                                                                   \
    template int cholesky_inplace<T>(Array<T> &in, const bool is_upper);                    \
    template Array<T> cholesky<T>   (int *info, const Array<T> &in, const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}

#endif
