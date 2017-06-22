/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(WITH_GRAPHICS)
#if defined(OS_WIN)
#include <windows.h>
#endif

// cuda_gl_interop.h does not include OpenGL headers for ARM
#include <common/graphics_common.hpp>
using namespace gl;
#define GL_VERSION gl::GL_VERSION
#define __gl_h_ //FIXME Hack to avoid gl.h inclusion by cuda_gl_interop.h
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <common/InteropManager.hpp>
#include <map>
#include <vector>

namespace cuda
{
typedef cudaGraphicsResource_t CGR_t;
typedef std::shared_ptr<CGR_t> SharedResource;
typedef std::vector<SharedResource> ShrdResVector;

class GraphicsResourceManager : public common::InteropManager<GraphicsResourceManager, CGR_t>
{
    public:
        GraphicsResourceManager() {}
        ShrdResVector registerResources(std::vector<uint32_t> resources);

    protected:
        GraphicsResourceManager(GraphicsResourceManager const&);
        void operator=(GraphicsResourceManager const&);
};
}
#endif
