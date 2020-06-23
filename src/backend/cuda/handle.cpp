/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/unique_handle.hpp>
#include <cublas.hpp>
#include <cufft.hpp>
#include <cusolverDn.hpp>
#include <cusparse.hpp>

// clang-format off
CREATE_HANDLE(cusparseMatDescr_t, cusparseCreateMatDescr, cusparseDestroyMatDescr);
CREATE_HANDLE(cusparseHandle_t, cusparseCreate, cusparseDestroy);
CREATE_HANDLE(cublasHandle_t, cublasCreate, cublasDestroy);
CREATE_HANDLE(cusolverDnHandle_t, cusolverDnCreate, cusolverDnDestroy);
CREATE_HANDLE(cufftHandle, cufftCreate, cufftDestroy);

#if defined(AF_USE_NEW_CUSPARSE_API)
namespace common {

template<>
void handle_deleter<cusparseSpMatDescr_t>(cusparseSpMatDescr_t handle) noexcept {
    cusparseDestroySpMat(handle);
}

template<>
void handle_deleter<cusparseDnVecDescr_t>(cusparseDnVecDescr_t handle) noexcept {
    cusparseDestroyDnVec(handle);
}

template<>
void handle_deleter<cusparseDnMatDescr_t>(cusparseDnMatDescr_t handle) noexcept {
    cusparseDestroyDnMat(handle);
}

}  // namespace common
#endif

#ifdef WITH_CUDNN

#include <cudnn.hpp>
#include <cudnnModule.hpp>

CREATE_HANDLE(cudnnHandle_t, cuda::getCudnnPlugin().cudnnCreate, cuda::getCudnnPlugin().cudnnDestroy);
CREATE_HANDLE(cudnnTensorDescriptor_t, cuda::getCudnnPlugin().cudnnCreateTensorDescriptor, cuda::getCudnnPlugin().cudnnDestroyTensorDescriptor);
CREATE_HANDLE(cudnnFilterDescriptor_t, cuda::getCudnnPlugin().cudnnCreateFilterDescriptor, cuda::getCudnnPlugin().cudnnDestroyFilterDescriptor);
CREATE_HANDLE(cudnnConvolutionDescriptor_t, cuda::getCudnnPlugin().cudnnCreateConvolutionDescriptor, cuda::getCudnnPlugin().cudnnDestroyConvolutionDescriptor);

#endif

// clang-format on
