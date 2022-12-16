/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <common/half.hpp>
#include <common/unique_handle.hpp>
#include <cudnnModule.hpp>
#include <af/dim4.hpp>

// clang-format off
DEFINE_HANDLER(cudnnHandle_t, arrayfire::cuda::getCudnnPlugin().cudnnCreate, arrayfire::cuda::getCudnnPlugin().cudnnDestroy);

DEFINE_HANDLER(cudnnTensorDescriptor_t, arrayfire::cuda::getCudnnPlugin().cudnnCreateTensorDescriptor, arrayfire::cuda::getCudnnPlugin().cudnnDestroyTensorDescriptor);

DEFINE_HANDLER(cudnnFilterDescriptor_t, arrayfire::cuda::getCudnnPlugin().cudnnCreateFilterDescriptor, arrayfire::cuda::getCudnnPlugin().cudnnDestroyFilterDescriptor);

DEFINE_HANDLER(cudnnConvolutionDescriptor_t, arrayfire::cuda::getCudnnPlugin().cudnnCreateConvolutionDescriptor, arrayfire::cuda::getCudnnPlugin().cudnnDestroyConvolutionDescriptor);
// clang-format on

namespace arrayfire {
namespace cuda {

const char *errorString(cudnnStatus_t err);

#define CUDNN_CHECK(fn)                                                     \
    do {                                                                    \
        cudnnStatus_t _error = (fn);                                        \
        if (_error == CUDNN_STATUS_SUCCESS) {                               \
            break;                                                          \
        } else if (_error == CUDNN_STATUS_ALLOC_FAILED) {                   \
            AF_ERROR(                                                       \
                "CUDNN Error(CUDNN_STATUS_ALLOC_FAILED): Error allocating " \
                "for function all ",                                        \
                AF_ERR_NO_MEM);                                             \
        } else if (_error == CUDNN_STATUS_NOT_SUPPORTED) {                  \
            CUDA_NOT_SUPPORTED(                                             \
                "CUDNN Error(CUDNN_STATUS_NOT_SUPPORTED): This version of " \
                "CUDNN does not support the data type or the size of this " \
                "operation");                                               \
        } else {                                                            \
            char _err_msg[1024];                                            \
            snprintf(_err_msg, sizeof(_err_msg), "CUDNN Error(%s): \n",     \
                     errorString(_error));                                  \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                            \
        }                                                                   \
    } while (0)

/// Returns a cuDNN type based on the template parameter
template<typename T>
cudnnDataType_t getCudnnDataType();

void cudnnSet(cudnnTensorDescriptor_t desc, cudnnDataType_t cudnn_dtype,
              af::dim4 dims);

void cudnnSet(cudnnFilterDescriptor_t desc, cudnnDataType_t cudnn_dtype,
              af::dim4 dims);

// cuDNN Wrappers
//
// cuDNN deprecates and releases function names often between releases. in order
// to prevent locking arrayfire versions to specific cuDNN versions, we wrap all
// cuDNN calls so that the main codebase is not full of ifdefs. The Following
// functions are wrappers around cuDNN functions that abstract out the version
// differences between older versions of cuDNN.
//

cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc,
    int pad_h,     // zero-padding height
    int pad_w,     // zero-padding width
    int u,         // vertical filter stride
    int v,         // horizontal filter stride
    int upscalex,  // upscale the input in x-direction
    int upscaley,  // upscale the input in y-direction
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType);

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int k,
                                         int c, int h, int w);

cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, /* image data type */
    int n,                    /* number of inputs (batch size) */
    int c,                    /* number of input feature maps */
    int h,                    /* height of input section */
    int w);                   /* width of input section */

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes);

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[]);

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle,
                                                          int *count);

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    cudnnHandle_t handle, int *count);

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes);

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes);

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults);

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionFwdAlgo_t *algo);

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo);

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw);

}  // namespace cuda
}  // namespace arrayfire
