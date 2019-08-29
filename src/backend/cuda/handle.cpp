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
#include <cudnn.hpp>
#include <cusolverDn.hpp>
#include <cusparse.hpp>
#include <cufft.hpp>

// clang-format off
CREATE_HANDLE(cusparseMatDescr_t, cusparseCreateMatDescr, cusparseDestroyMatDescr);
CREATE_HANDLE(cusparseHandle_t, cusparseCreate, cusparseDestroy);
CREATE_HANDLE(cublasHandle_t, cublasCreate, cublasDestroy);
CREATE_HANDLE(cusolverDnHandle_t, cusolverDnCreate, cusolverDnDestroy);
CREATE_HANDLE(cufftHandle, cufftCreate, cufftDestroy);
CREATE_HANDLE(cudnnHandle_t, cudnnCreate, cudnnDestroy);
CREATE_HANDLE(cudnnTensorDescriptor_t, cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor);
CREATE_HANDLE(cudnnFilterDescriptor_t, cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor);
CREATE_HANDLE(cudnnConvolutionDescriptor_t, cudnnCreateConvolutionDescriptor, cudnnDestroyConvolutionDescriptor);


// clang-format on
