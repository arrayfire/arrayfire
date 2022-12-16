/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <GraphicsResourceManager.hpp>

#include <common/graphics_common.hpp>
// cuda_gl_interop.h does not include OpenGL headers for ARM
// __gl_h_ should be defined by glad.h inclusion
#include <cuda_gl_interop.h>
#include <err_cuda.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace cuda {
GraphicsResourceManager::ShrdResVector
GraphicsResourceManager::registerResources(
    const std::vector<uint32_t>& resources) {
    ShrdResVector output;

    auto deleter = [](cudaGraphicsResource_t* handle) {
        // FIXME Having a CUDA_CHECK around unregister
        // call is causing invalid GL context.
        // Moving ForgeManager class singleton as data
        // member of DeviceManager with proper ordering
        // of member destruction doesn't help either.
        // Calling makeContextCurrent also doesn't help.
        cudaGraphicsUnregisterResource(*handle);
        delete handle;
    };

    for (auto id : resources) {
        cudaGraphicsResource_t r;
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &r, id, cudaGraphicsMapFlagsWriteDiscard));
        output.emplace_back(new cudaGraphicsResource_t(r), deleter);
    }

    return output;
}
}  // namespace cuda
}  // namespace arrayfire
