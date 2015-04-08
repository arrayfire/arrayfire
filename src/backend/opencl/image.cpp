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
#include <Array.hpp>
#include <image.hpp>
#include <platform.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <stdexcept>
#include <cstdio>
#include <map>
#include <vector>

namespace opencl
{

typedef std::map<fg_image_handle, cl::Buffer*> interop_t;
typedef interop_t::iterator iter_t;

class InteropManager
{
    private:
        interop_t interop_maps[DeviceManager::MAX_DEVICES];

    public:
        static InteropManager& getInstance();
        ~InteropManager();
        cl::Buffer* getPBOResource(const fg_image_handle image);

    protected:
        InteropManager() {}
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);
        void destroyResources();
};

void InteropManager::destroyResources()
{
    int n = getActiveDeviceId();
    for(iter_t iter = interop_maps[n].begin(); iter != interop_maps[n].end(); iter++)
        delete iter->second;
}

InteropManager::~InteropManager()
{
    for(int i = 0; i < getDeviceCount(); i++) {
        setDevice(i);
        destroyResources();
    }
}

InteropManager& InteropManager::getInstance()
{
    static InteropManager my_instance;
    return my_instance;
}

cl::Buffer* InteropManager::getPBOResource(const fg_image_handle image)
{
    int device = getActiveDeviceId();
    iter_t iter = interop_maps[device].find(image);

    if (iter == interop_maps[device].end())
        interop_maps[device][image] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, image->gl_PBO, NULL);

    return interop_maps[device][image];
}

template<typename T>
void copy_image(const Array<T> &in, const fg_image_handle image)
{
    InteropManager& intrpMngr = InteropManager::getInstance();

    cl::Buffer *clPBOResource = intrpMngr.getPBOResource(image);
    const cl::Buffer *d_X = in.get();
    size_t num_bytes = in.elements()*sizeof(T);

    std::vector<cl::Memory> shared_objects;
    shared_objects.push_back(*clPBOResource);

    glFinish();
    getQueue().enqueueAcquireGLObjects(&shared_objects);
    getQueue().enqueueCopyBuffer(*d_X, *clPBOResource, 0, 0, num_bytes, NULL, NULL);
    getQueue().finish();
    getQueue().enqueueReleaseGLObjects(&shared_objects);

    CheckGL("After opencl resource copy");
    CL_DEBUG_FINISH(getQueue());
}

#define INSTANTIATE(T)      \
    template void copy_image<T>(const Array<T> &in, const fg_image_handle image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)

}

#endif
