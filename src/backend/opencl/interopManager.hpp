/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <GL/glew.h>
#include <graphics_common.hpp>
#include <platform.hpp>
#include <map>

namespace opencl
{

typedef std::map<void *, cl::Buffer*> interop_t;
typedef interop_t::iterator iter_t;

class InteropManager
{
    private:
        interop_t interop_maps[DeviceManager::MAX_DEVICES];

    public:
        static InteropManager& getInstance();
        ~InteropManager();
        cl::Buffer* getBufferResource(const fg::Image* image);
        cl::Buffer* getBufferResource(const fg::Plot* plot);
        cl::Buffer* getBufferResource(const fg::Histogram* hist);

    protected:
        InteropManager() {}
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);
        void destroyResources();
};

}

#endif
