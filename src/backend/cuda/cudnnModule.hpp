/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/DependencyModule.hpp>

#include <cudnn.h>

#include <memory>
#include <tuple>

#if CUDNN_VERSION > 4000
// This function is not available on versions greater than v4
cudnnStatus_t cudnnSetFilter4dDescriptor_v4(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,  // image data type
    cudnnTensorFormat_t format,
    int k,   // number of output feature maps
    int c,   // number of input feature maps
    int h,   // height of each input filter
    int w);  // width of  each input filter
#else
// This function is only available on newer versions of cudnn
size_t cudnnGetCudartVersion(void);
#endif

namespace cuda {

class cudnnModule {
    common::DependencyModule module;
    int major, minor, patch;

   public:
    cudnnModule();
    MODULE_MEMBER(cudnnConvolutionBackwardData);
    MODULE_MEMBER(cudnnConvolutionBackwardFilter);
    MODULE_MEMBER(cudnnConvolutionForward);
    MODULE_MEMBER(cudnnCreate);
    MODULE_MEMBER(cudnnCreateConvolutionDescriptor);
    MODULE_MEMBER(cudnnCreateFilterDescriptor);
    MODULE_MEMBER(cudnnCreateTensorDescriptor);
    MODULE_MEMBER(cudnnDestroy);
    MODULE_MEMBER(cudnnDestroyConvolutionDescriptor);
    MODULE_MEMBER(cudnnDestroyFilterDescriptor);
    MODULE_MEMBER(cudnnDestroyTensorDescriptor);
    MODULE_MEMBER(cudnnGetConvolutionBackwardDataWorkspaceSize);
    MODULE_MEMBER(cudnnGetConvolutionBackwardFilterAlgorithm);
    MODULE_MEMBER(cudnnGetConvolutionBackwardFilterWorkspaceSize);
    MODULE_MEMBER(cudnnGetConvolutionForwardAlgorithm);
    MODULE_MEMBER(cudnnGetConvolutionForwardWorkspaceSize);
    MODULE_MEMBER(cudnnGetConvolutionNdForwardOutputDim);
    MODULE_MEMBER(cudnnSetConvolution2dDescriptor);
    MODULE_MEMBER(cudnnSetFilter4dDescriptor);
    MODULE_MEMBER(cudnnSetFilter4dDescriptor_v4);
    MODULE_MEMBER(cudnnGetVersion);
    MODULE_MEMBER(cudnnGetCudartVersion);
    MODULE_MEMBER(cudnnSetStream);
    MODULE_MEMBER(cudnnSetTensor4dDescriptor);

    spdlog::logger* getLogger() const noexcept;

    /// Returns the version of the cuDNN loaded at runtime
    std::tuple<int, int, int> getVersion() const noexcept {
        return std::make_tuple(major, minor, patch);
    }

    bool isLoaded() const noexcept { return module.isLoaded(); }
};

cudnnModule& getCudnnPlugin() noexcept;

}  // namespace cuda
