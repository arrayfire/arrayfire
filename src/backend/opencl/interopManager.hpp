/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>

#include <graphics_common.hpp>
#include <platform.hpp>

#include <map>
#include <vector>

namespace cl
{
class Buffer;
}

namespace opencl
{

typedef std::map<void *, std::vector<cl::Buffer*> > interop_t;
typedef interop_t::iterator iter_t;

class InteropManager
{
    private:
        interop_t interop_maps[DeviceManager::MAX_DEVICES];

    public:
        InteropManager() {}
        ~InteropManager();
        cl::Buffer** getBufferResource(const forge::Image        *handle);
        cl::Buffer** getBufferResource(const forge::Plot         *handle);
        cl::Buffer** getBufferResource(const forge::Histogram    *handle);
        cl::Buffer** getBufferResource(const forge::Surface      *handle);
        cl::Buffer** getBufferResource(const forge::VectorField  *handle);

    protected:
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);
        interop_t& getDeviceMap(int device = -1); // default will return current device
        void destroyResources();
};

}

#endif
