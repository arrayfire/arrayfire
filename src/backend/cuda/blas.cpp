/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <platform.hpp>

#include <stdexcept>
#include <string>
#include <cassert>
#include <iostream>
#include <math.hpp>
#include <err_common.hpp>
#include <boost/scoped_ptr.hpp>

namespace cuda
{

static
const char * const
cublasErrorString(cublasStatus_t err)
{

    switch(err)
    {
        case    CUBLAS_STATUS_SUCCESS:              return "CUBLAS_STATUS_SUCCESS";
        case    CUBLAS_STATUS_NOT_INITIALIZED:      return "CUBLAS_STATUS_NOT_INITIALIZED";
        case    CUBLAS_STATUS_ALLOC_FAILED:         return "CUBLAS_STATUS_ALLOC_FAILED";
        case    CUBLAS_STATUS_INVALID_VALUE:        return "CUBLAS_STATUS_INVALID_VALUE";
        case    CUBLAS_STATUS_ARCH_MISMATCH:        return "CUBLAS_STATUS_ARCH_MISMATCH";
        case    CUBLAS_STATUS_MAPPING_ERROR:        return "CUBLAS_STATUS_MAPPING_ERROR";
        case    CUBLAS_STATUS_EXECUTION_FAILED:     return "CUBLAS_STATUS_EXECUTION_FAILED";
        case    CUBLAS_STATUS_INTERNAL_ERROR:       return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION > 5050
        case    CUBLAS_STATUS_NOT_SUPPORTED:        return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
        default:                                    return "UNKNOWN";
    }
}

//RAII class around the cublas Handle
class cublasHandle
{
    cublasHandle_t handle;
public:
    cublasHandle() : handle(0){
        cublasStatus_t cErr;
        cErr = cublasCreate(&handle);
        if(cErr != CUBLAS_STATUS_SUCCESS) {
            using std::string;
            throw std::runtime_error(string("cuBLAS Error: ") + cublasErrorString(cErr));
        }
    }

    ~cublasHandle()             { cublasDestroy(handle);}
    operator cublasHandle_t()   { return handle;        }
};

cublasHandle&
getHandle()
{
    using boost::scoped_ptr;
    static scoped_ptr<cublasHandle> handle[DeviceManager::MAX_DEVICES];
    if(!handle[getActiveDeviceId()]) {
        handle[getActiveDeviceId()].reset(new cublasHandle());
    }

    return *handle[getActiveDeviceId()];
}

cublasOperation_t
toCblasTranspose(af_blas_transpose opt)
{
    cublasOperation_t out = CUBLAS_OP_N;
    switch(opt) {
        case AF_NO_TRANSPOSE        : out = CUBLAS_OP_N;    break;
        case AF_TRANSPOSE           : out = CUBLAS_OP_T;    break;
        case AF_CONJUGATE_TRANSPOSE : out = CUBLAS_OP_C;    break;
        default                     : AF_ERROR("INVALID af_blas_transpose", AF_ERR_INVALID_ARG);
    }
    return out;
}

template<typename T>
struct gemm_func_def_t
{
    typedef cublasStatus_t (*gemm_func_def)(    cublasHandle_t,
                                                cublasOperation_t, cublasOperation_t,
                                                int, int, int,
                                                const T *,  const T *, int,
                                                            const T *, int,
                                                const T *,        T *, int);
};

template<typename T>
struct gemv_func_def_t
{
    typedef cublasStatus_t (*gemv_func_def)(    cublasHandle_t,
                                                cublasOperation_t,
                                                int, int,
                                                const T *,  const T *, int,
                                                            const T *, int,
                                                const T *,        T *, int);
};

template<typename T>
struct dot_func_def_t
{
    typedef cublasStatus_t (*dot_func_def)(    cublasHandle_t,
                                                int,
                                                const T *,  int,
                                                const T *,  int,
                                                T *);
};

#define BLAS_FUNC_DEF( FUNC )                       \
template<typename T>                                \
typename FUNC##_func_def_t<T>::FUNC##_func_def      \
FUNC##_func();

#define BLAS_FUNC( FUNC, TYPE, PREFIX )         \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def       FUNC##_func<TYPE>()  { return &cublas##PREFIX##FUNC; }

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float,  S)
BLAS_FUNC(gemm, cfloat, C)
BLAS_FUNC(gemm, double, D)
BLAS_FUNC(gemm, cdouble,Z)

BLAS_FUNC_DEF(gemv)
BLAS_FUNC(gemv, float,  S)
BLAS_FUNC(gemv, cfloat, C)
BLAS_FUNC(gemv, double, D)
BLAS_FUNC(gemv, cdouble,Z)

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, float,  S)
BLAS_FUNC(dot, double, D)

using namespace std;

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    cublasOperation_t lOpts = toCblasTranspose(optLhs);
    cublasOperation_t rOpts = toCblasTranspose(optRhs);

    int aRowDim = (lOpts == CUBLAS_OP_N) ? 0 : 1;
    int aColDim = (lOpts == CUBLAS_OP_N) ? 1 : 0;
    int bColDim = (rOpts == CUBLAS_OP_N) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[aRowDim];
    int N = rDims[bColDim];
    int K = lDims[aColDim];

    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    T alpha = scalar<T>(1);
    T beta  = scalar<T>(0);

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    if(rDims[bColDim] == 1) {
        N = lDims[aColDim];
        gemv_func<T>()(
            getHandle(), lOpts,
            lDims[0], lDims[1],
            &alpha, lhs.get(),   lStrides[1],
                    rhs.get(),   rStrides[0],
            &beta , out.get(),            1);
    } else {
        cublasStatus_t err =
            gemm_func<T>()(
                getHandle(), lOpts, rOpts,
                M, N, K,
                &alpha, lhs.get(),  lStrides[1],
                        rhs.get(),  rStrides[1],
                &beta , out.get(), out.dims()[0]);

        if(err != CUBLAS_STATUS_SUCCESS) {
            std::cout <<__PRETTY_FUNCTION__<< " ERROR: " << cublasErrorString(err) << std::endl;
        }
    }

    return out;

}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs,
             af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    int N = lhs.dims()[0];

    T out;
    dot_func<T>()(  getHandle(), N,
                            lhs.get(), lhs.strides()[0],
                            rhs.get(), rhs.strides()[0],
                            &out);

    return createValueArray(af::dim4(1), out);
}

#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,  \
                                      af_blas_transpose optLhs, af_blas_transpose optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                       \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                                   af_blas_transpose optLhs, af_blas_transpose optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
}
