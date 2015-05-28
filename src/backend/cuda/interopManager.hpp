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
        ~InteropManager();
        cudaGraphicsResource* getBufferResource(const fg::Image* handle);
        cudaGraphicsResource* getBufferResource(const fg::Plot* handle);
        cudaGraphicsResource* getBufferResource(const fg::Histogram* handle);

    protected:
        InteropManager() {}
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);
        void destroyResources();
};

}

#endif
