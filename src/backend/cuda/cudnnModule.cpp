/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/Logger.hpp>
#include <common/err_common.hpp>
#include <common/util.hpp>
#include <cudnnModule.hpp>
#include <device_manager.hpp>

#include <string>
#include <tuple>

using std::string;

namespace cuda {

spdlog::logger* cudnnModule::getLogger() { return module.getLogger(); }

auto cudnnVersionComponents(size_t version) {
    int major = version / 1000;
    int minor = (version - (major * 1000)) / 100;
    int patch = (version - (major * 1000) - (minor * 100));
    return std::tuple<int, int, int>(major, minor, patch);
}

cudnnModule::cudnnModule()
    : module({"cudnn"}, {"", "64_7", "64_8", "64_6", "64_5", "64_4"}, {""}) {
    if (!module.isLoaded()) {
        string error_message =
            "Error loading cuDNN: " + module.getErrorMessage() +
            "\ncuDNN failed to load. Try installing cuDNN or check if cuDNN is "
            "in the search path. On Linux, you can set the LD_DEBUG=libs "
            "environment variable to debug loading issues.";
        AF_ERROR(error_message.c_str(), AF_ERR_LOAD_LIB);
    }

    MODULE_FUNCTION_INIT(cudnnGetVersion);

    int rtmajor, rtminor;
    int cudnn_version             = this->cudnnGetVersion();
    int cudnn_rtversion           = 0;
    std::tie(major, minor, patch) = cudnnVersionComponents(cudnn_version);

    if (cudnn_version >= 6000) {
        MODULE_FUNCTION_INIT(cudnnGetCudartVersion);
        cudnn_rtversion = this->cudnnGetCudartVersion();
    } else {
        AF_TRACE(
            "Warning: This version of cuDNN({}.{}) does not support "
            "cudnnGetCudartVersion. No runtime checks performed.",
            major, minor);
    }

    std::tie(rtmajor, rtminor, std::ignore) =
        cudnnVersionComponents(cudnn_rtversion);

    AF_TRACE("cuDNN Version: {}.{}.{} cuDNN CUDA Runtime: {}.{}", major, minor,
             patch, rtmajor, rtminor);

    // Check to see if the version of cuDNN ArrayFire was compiled against
    // is compatible with the version loaded at runtime
    if (CUDNN_VERSION <= 6000 && cudnn_version > CUDNN_VERSION) {
        string error_msg = fmt::format(
            "ArrayFire was compiled with an older version of cuDNN({}.{}) that "
            "does not support the version that was loaded at runtime({}.{}).",
            CUDNN_MAJOR, CUDNN_MINOR, major, minor);
        AF_ERROR(error_msg, AF_ERR_NOT_SUPPORTED);
    }

    int afcuda_runtime = 0;
    cudaRuntimeGetVersion(&afcuda_runtime);
    if (afcuda_runtime != cudnn_version) {
        getLogger()->warn(
            "WARNING: ArrayFire CUDA Runtime({}) and cuDNN CUDA "
            "Runtime({}.{}) do not match. For maximum compatibility, make sure "
            "the two versions match.(Ignoring check)",
            // NOTE: the int version formats from CUDA and cuDNN are different
            // so we are using int_version_to_string for the ArrayFire CUDA
            // runtime
            int_version_to_string(afcuda_runtime), rtmajor, rtminor);
    }

    MODULE_FUNCTION_INIT(cudnnConvolutionBackwardData);
    MODULE_FUNCTION_INIT(cudnnConvolutionBackwardFilter);
    MODULE_FUNCTION_INIT(cudnnConvolutionForward);
    MODULE_FUNCTION_INIT(cudnnCreate);
    MODULE_FUNCTION_INIT(cudnnCreateConvolutionDescriptor);
    MODULE_FUNCTION_INIT(cudnnCreateFilterDescriptor);
    MODULE_FUNCTION_INIT(cudnnCreateTensorDescriptor);
    MODULE_FUNCTION_INIT(cudnnDestroy);
    MODULE_FUNCTION_INIT(cudnnDestroyConvolutionDescriptor);
    MODULE_FUNCTION_INIT(cudnnDestroyFilterDescriptor);
    MODULE_FUNCTION_INIT(cudnnDestroyTensorDescriptor);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionBackwardDataWorkspaceSize);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionBackwardFilterAlgorithm);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionBackwardFilterWorkspaceSize);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionForwardAlgorithm);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionForwardWorkspaceSize);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionNdForwardOutputDim);
    MODULE_FUNCTION_INIT(cudnnSetConvolution2dDescriptor);
    MODULE_FUNCTION_INIT(cudnnSetFilter4dDescriptor);
    if (major == 4) { MODULE_FUNCTION_INIT(cudnnSetFilter4dDescriptor_v4); }
    MODULE_FUNCTION_INIT(cudnnSetStream);
    MODULE_FUNCTION_INIT(cudnnSetTensor4dDescriptor);

    // Check to see if the cuDNN runtime is compatible with the current device
    cudaDeviceProp prop = getDeviceProp(getActiveDeviceId());
    if (!checkDeviceWithRuntime(cudnn_rtversion, {prop.major, prop.minor})) {
        string error_message = fmt::format(
            "Error: cuDNN CUDA Runtime({}.{}) does not support the "
            "current device's compute capability(sm_{}{}).",
            rtmajor, rtminor, prop.major, prop.minor);
        AF_ERROR(error_message, AF_ERR_RUNTIME);
    }

    if (!module.symbolsLoaded()) {
        string error_message =
            "Error loading cuDNN symbols. ArrayFire was unable to load some "
            "symbols from the cuDNN library. Please create an issue on the "
            "ArrayFire repository with information about the installed cuDNN "
            "and ArrayFire on your system.";
        AF_ERROR(error_message, AF_ERR_LOAD_LIB);
    }
}

cudnnModule& getCudnnPlugin() {
    static cudnnModule* plugin = new cudnnModule();
    return *plugin;
}

}  // namespace cuda
