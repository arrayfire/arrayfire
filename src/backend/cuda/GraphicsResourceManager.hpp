/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)
#if defined(OS_WIN)
#include <windows.h>
#endif

#include <platform.hpp>
// cuda_gl_interop.h does not include OpenGL headers for ARM
#include <graphics_common.hpp>
using namespace gl;
#define GL_VERSION gl::GL_VERSION
#define __gl_h_ //Hack to avoid gl.h inclusion by cuda_gl_interop.h
#include <cuda_gl_interop.h>
#include <common/InteropManager.hpp>
#include <err_cuda.hpp>
#include <map>
#include <vector>

namespace cuda
{
typedef cudaGraphicsResource_t CGR_t;

class GraphicsResourceManager : public common::InteropManager<GraphicsResourceManager, CGR_t>
{
    public:
        GraphicsResourceManager() {}

        std::vector<CGR_t> registerResources(std::vector<uint32_t> resources) {
            std::vector<CGR_t> output;

            for (auto id: resources) {
                CGR_t r;
                CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&r, id, cudaGraphicsMapFlagsWriteDiscard));
                output.push_back(r);
            }

            return output;
        }

        void unregisterResource(CGR_t handle) {
            CUDA_CHECK(cudaGraphicsUnregisterResource(handle));
        }

    protected:
        GraphicsResourceManager(GraphicsResourceManager const&);
        void operator=(GraphicsResourceManager const&);
};
}
#endif
