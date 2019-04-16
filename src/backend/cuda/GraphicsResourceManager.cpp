/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(OS_WIN)
#include <windows.h>
#endif

// cuda_gl_interop.h does not include OpenGL headers for ARM
#include <common/graphics_common.hpp>
#define __gl_h_  // FIXME Hack to avoid gl.h inclusion by cuda_gl_interop.h
#include <GraphicsResourceManager.hpp>
#include <cuda_gl_interop.h>
#include <err_cuda.hpp>
#include <platform.hpp>

namespace cuda {
GraphicsResourceManager::ShrdResVector
GraphicsResourceManager::registerResources(std::vector<uint32_t> resources) {
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
