/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#if defined(WITH_GRAPHICS)

// cuda_gl_interop.h does not include OpenGL headers for ARM
#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>

#if defined(__arm__) || defined(__aarch64__)
using namespace gl;
#define GL_VERSION gl::GL_VERSION
#endif

#include <platform.hpp>
#include <graphics_common.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <map>

using af::dim4;

namespace cuda
{

typedef std::map<void *, cudaGraphicsResource *> interop_t;
typedef interop_t::iterator iter_t;

// Manager Class for cudaPBOResource: calls garbage collection at the end of the program
class InteropManager
{
    private:
        interop_t interop_maps[DeviceManager::MAX_DEVICES];

    public:
        static InteropManager& getInstance();
        static bool checkGraphicsInteropCapability();

        ~InteropManager();
        cudaGraphicsResource* getBufferResource(const forge::Image* handle);
        cudaGraphicsResource* getBufferResource(const forge::Plot* handle);
        cudaGraphicsResource* getBufferResource(const forge::Histogram* handle);
        cudaGraphicsResource* getBufferResource(const forge::Surface* handle);

    protected:
        InteropManager() {}
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);
        void destroyResources();
};

}

#endif
