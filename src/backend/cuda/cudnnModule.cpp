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
#include <utility.hpp>

#include <array>
#include <string>
#include <tuple>

using arrayfire::common::int_version_to_string;
using arrayfire::common::Version;
using std::make_tuple;
using std::string;

namespace arrayfire {
namespace cuda {

// clang-format off
// Latest version from each minor releases are enlisted below
constexpr std::array<common::Version, 11> cudnnVersions = {
    make_tuple(8, 0,  1),
    make_tuple(7, 6,  5),
    make_tuple(7, 5,  1),
    make_tuple(7, 4,  2),
    make_tuple(7, 3,  1),
    make_tuple(7, 2,  1),
    make_tuple(7, 1,  4),
    make_tuple(7, 0,  5),
    make_tuple(6, 0, 21),
    make_tuple(5, 1, 10),
    make_tuple(4, 0,  7)
};
// clang-format on

spdlog::logger* cudnnModule::getLogger() const noexcept {
    return module.getLogger();
}

auto cudnnVersionComponents(size_t version) {
    size_t major = version / 1000;
    size_t minor = (version - (major * 1000)) / 100;
    size_t patch = (version - (major * 1000) - (minor * 100));
    return make_tuple(major, minor, patch);
}

auto cudaRuntimeVersionComponents(size_t version) {
    auto major = version / 1000;
    auto minor = (version - (major * 1000)) / 10;
    return make_tuple(major, minor);
}

cudnnModule::cudnnModule()
    : module({"cudnn"}, {"", "64_7", "64_8", "64_6", "64_5", "64_4"}, {""},
             cudnnVersions.size(), cudnnVersions.data()) {
    if (!module.isLoaded()) {
        AF_TRACE(
            "WARNING: Unable to load cuDNN: {}"
            "\ncuDNN failed to load. Try installing cuDNN or check if cuDNN is "
            "in the search path. On Linux, you can set the LD_DEBUG=libs "
            "environment variable to debug loading issues. Falling back to "
            "matmul based implementation",
            module.getErrorMessage());

        return;
    }

    MODULE_FUNCTION_INIT(cudnnGetVersion);

    int rtmajor, rtminor;
    size_t cudnn_version          = this->cudnnGetVersion();
    size_t cudnn_rtversion        = 0;
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

    std::tie(rtmajor, rtminor) = cudaRuntimeVersionComponents(cudnn_rtversion);

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
    if (afcuda_runtime != static_cast<int>(cudnn_rtversion)) {
        getLogger()->warn(
            "WARNING: ArrayFire CUDA Runtime({}) and cuDNN CUDA "
            "Runtime({}) do not match. For maximum compatibility, make sure "
            "the two versions match.(Ignoring check)",
            // NOTE: the int version formats from CUDA and cuDNN are different
            // so we are using int_version_to_string for the ArrayFire CUDA
            // runtime
            int_version_to_string(afcuda_runtime),
            int_version_to_string(cudnn_rtversion));
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
    MODULE_FUNCTION_INIT(cudnnGetConvolutionForwardAlgorithmMaxCount);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionForwardWorkspaceSize);
    MODULE_FUNCTION_INIT(cudnnGetConvolutionBackwardFilterWorkspaceSize);
    MODULE_FUNCTION_INIT(cudnnFindConvolutionForwardAlgorithm);
    MODULE_FUNCTION_INIT(cudnnFindConvolutionBackwardFilterAlgorithm);
    if (major < 8) {
        MODULE_FUNCTION_INIT(cudnnGetConvolutionForwardAlgorithm);
        MODULE_FUNCTION_INIT(cudnnGetConvolutionBackwardFilterAlgorithm);
    }
    MODULE_FUNCTION_INIT(cudnnGetConvolutionNdForwardOutputDim);
    MODULE_FUNCTION_INIT(cudnnSetConvolution2dDescriptor);
    MODULE_FUNCTION_INIT(cudnnSetFilter4dDescriptor);
    if (major == 4) { MODULE_FUNCTION_INIT(cudnnSetFilter4dDescriptor_v4); }
    MODULE_FUNCTION_INIT(cudnnSetStream);
    MODULE_FUNCTION_INIT(cudnnSetTensor4dDescriptor);

    if (!module.symbolsLoaded()) {
        string error_message =
            "Error loading cuDNN symbols. ArrayFire was unable to load some "
            "symbols from the cuDNN library. Please create an issue on the "
            "ArrayFire repository with information about the installed cuDNN "
            "and ArrayFire on your system.";
        AF_ERROR(error_message, AF_ERR_LOAD_LIB);
    }
}

cudnnModule& getCudnnPlugin() noexcept {
    static auto* plugin = new cudnnModule();
    return *plugin;
}

}  // namespace cuda
}  // namespace arrayfire
