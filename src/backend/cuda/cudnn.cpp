/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cudnn.hpp>
#include <err_cuda.hpp>

namespace cuda {

const char *errorString(cudnnStatus_t err) {
    switch (err) {
        case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED:
            return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED:
            return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION >= 6000
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#if CUDNN_VERSION >= 7000
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
#endif
#endif
        default: return "UNKNOWN";
    }
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc,
    int pad_h,     // zero-padding height
    int pad_w,     // zero-padding width
    int u,         // vertical filter stride
    int v,         // horizontal filter stride
    int upscalex,  // upscale the input in x-direction
    int upscaley,  // upscale the input in y-direction
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
    return
#if CUDNN_VERSION >= 6000
        getCudnnPlugin().cudnnSetConvolution2dDescriptor(
            convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode,
            computeType);
#elif CUDNN_VERSION >= 4000
        getCudnnPlugin().cudnnSetConvolution2dDescriptor(
            convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode);
#else
        static_assert(1 != 1, "cuDNN version not supported");
#endif
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int k,
                                         int c, int h, int w) {
#if CUDNN_VERSION >= 6000
    int version = getCudnnPlugin().cudnnGetVersion();
    if (version >= 6000) {
        return getCudnnPlugin().cudnnSetFilter4dDescriptor(filterDesc, dataType,
                                                           format, k, c, h, w);
    } else if (version == 4000) {
        return getCudnnPlugin().cudnnSetFilter4dDescriptor_v4(
            filterDesc, dataType, format, k, c, h, w);
    }
    CUDA_NOT_SUPPORTED(
        "cudnnSetFilter4dDescriptor not supported for the current version of "
        "cuDNN");
#elif CUDNN_VERSION == 4000
    return getCudnnPlugin().cudnnSetFilter4dDescriptor_v4(filterDesc, dataType,
                                                          format, k, c, h, w);
#else
    static_assert(1 != 1, "cuDNN version not supported");
#endif
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnTensorFormat_t format,
                                         cudnnDataType_t dataType, int n, int c,
                                         int h, int w) {
    return getCudnnPlugin().cudnnSetTensor4dDescriptor(tensorDesc, format,
                                                       dataType, n, c, h, w);
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
    return getCudnnPlugin().cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    return getCudnnPlugin().cudnnConvolutionBackwardData(
        handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, dxDesc, dx);
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[]) {
    return getCudnnPlugin().cudnnGetConvolutionNdForwardOutputDim(
        convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionFwdAlgo_t *algo) {
    return getCudnnPlugin().cudnnGetConvolutionForwardAlgorithm(
        handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes,
        algo);
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
    return getCudnnPlugin().cudnnGetConvolutionForwardWorkspaceSize(
        handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
}

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
    return getCudnnPlugin().cudnnConvolutionForward(
        handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, yDesc, y);
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
    return getCudnnPlugin().cudnnGetConvolutionBackwardFilterAlgorithm(
        handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes,
        algo);
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
    return getCudnnPlugin().cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
}

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {
    return getCudnnPlugin().cudnnConvolutionBackwardFilter(
        handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
        workSpaceSizeInBytes, beta, dwDesc, dw);
}

}  // namespace cuda
