/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cudnnModule.hpp>

#include <common/ArrayFireTypesIO.hpp>
#include <common/Logger.hpp>
#include <common/err_common.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <utility.hpp>

#include <array>
#include <string>
#include <tuple>

using arrayfire::common::fromCudaVersion;
using arrayfire::common::Version;
using std::make_tuple;
using std::string;

namespace arrayfire {
namespace cuda {

// clang-format off
// Latest version from each minor releases are enlisted below
constexpr std::array<common::Version, 11> cudnnVersions = {
    Version(8, 0,  1),
    Version(7, 6,  5),
    Version(7, 5,  1),
    Version(7, 4,  2),
    Version(7, 3,  1),
    Version(7, 2,  1),
    Version(7, 1,  4),
    Version(7, 0,  5),
    Version(6, 0, 21),
    Version(5, 1, 10),
    Version(4, 0,  7)
};
// clang-format on

spdlog::logger* cudnnModule::getLogger() const noexcept {
    return module.getLogger();
}

Version cudnnVersionComponents(size_t version) {
    int major = static_cast<int>(version / 1000);
    int minor = static_cast<int>((version - (major * 1000)) / 100);
    int patch = static_cast<int>(version - (major * 1000) - (minor * 100));
    return {major, minor, patch};
}

Version cudaRuntimeVersionComponents(size_t version) {
    int major = static_cast<int>(version / 1000);
    int minor = static_cast<int>((version - (major * 1000)) / 10);
    int patch =
        static_cast<int>((version - (major * 1000) - (minor * 10)) / 10);
    return {major, minor, patch};
}

Version getCudnnVersion(const LibHandle& handle) {
    std::function<size_t()> fptr(reinterpret_cast<size_t (*)()>(
        common::getFunctionPointer(handle, "cudnnGetVersion")));
    size_t v = fptr();

    return cudnnVersionComponents(v);
}

cudnnModule::cudnnModule()
    : module({"cudnn"}, {"", "64_8", "64_7", "64_6", "64_5", "64_4"}, {""},
             cudnnVersions.size(), cudnnVersions.data(), getCudnnVersion) {
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

    size_t cudnn_rtversion_val = 0;

    Version cudnn_version = module.getVersion();
    if (cudnn_version < Version(6)) {
        AF_TRACE(
            "Warning: This version of cuDNN({}) does not support "
            "cudnnGetCudartVersion. No runtime checks performed.",
            cudnn_version);
    } else {
        MODULE_FUNCTION_INIT(cudnnGetCudartVersion);
        cudnn_rtversion_val = this->cudnnGetCudartVersion();
    }

    Version cudnn_rtversion = cudaRuntimeVersionComponents(cudnn_rtversion_val);

    AF_TRACE("cuDNN Version: {} cuDNN CUDA Runtime: {}", cudnn_version,
             cudnn_rtversion);

    Version compiled_cudnn_version = fromCudaVersion(CUDNN_VERSION);

    // Check to see if the version of cuDNN ArrayFire was compiled against
    // is compatible with the version loaded at runtime
    if (compiled_cudnn_version.major <= 6 &&
        compiled_cudnn_version < cudnn_version) {
        string error_msg = fmt::format(
            "ArrayFire was compiled with an older version of cuDNN({}.{}) that "
            "does not support the version that was loaded at runtime({}.{}).",
            CUDNN_MAJOR, CUDNN_MINOR, cudnn_version.major, cudnn_version.minor);
        AF_ERROR(error_msg, AF_ERR_NOT_SUPPORTED);
    }

    int afcuda_runtime_version = 0;
    cudaRuntimeGetVersion(&afcuda_runtime_version);
    Version afcuda_runtime = fromCudaVersion(afcuda_runtime_version);
    if (afcuda_runtime != cudnn_rtversion) {
        getLogger()->warn(
            "WARNING: ArrayFire CUDA Runtime({}) and cuDNN CUDA "
            "Runtime({}) do not match. For maximum compatibility, make sure "
            "the two versions match.(Ignoring check)",
            // NOTE: the int version formats from CUDA and cuDNN are different
            // so we are using int_version_to_string for the ArrayFire CUDA
            // runtime
            afcuda_runtime, cudnn_rtversion);
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
    if (cudnn_version.major < 8) {
        MODULE_FUNCTION_INIT(cudnnGetConvolutionForwardAlgorithm);
        MODULE_FUNCTION_INIT(cudnnGetConvolutionBackwardFilterAlgorithm);
    }
    MODULE_FUNCTION_INIT(cudnnGetConvolutionNdForwardOutputDim);
    MODULE_FUNCTION_INIT(cudnnSetConvolution2dDescriptor);
    MODULE_FUNCTION_INIT(cudnnSetFilter4dDescriptor);
    if (cudnn_version.major == 4) {
        MODULE_FUNCTION_INIT(cudnnSetFilter4dDescriptor_v4);
    }
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
