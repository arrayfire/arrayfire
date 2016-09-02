/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <interopManager.hpp>

namespace opencl
{

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

cl::Buffer* InteropManager::getBufferResource(const forge::Image* image)
{
    void * key = (void*)image;
    int device = getActiveDeviceId();
    iter_t iter = interop_maps[device].find(key);

    if (iter == interop_maps[device].end())
        interop_maps[device][key] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, image->pbo(), NULL);

    return interop_maps[device][key];
}

cl::Buffer* InteropManager::getBufferResource(const forge::Plot* plot)
{
    void * key = (void*)plot;
    int device = getActiveDeviceId();
    iter_t iter = interop_maps[device].find(key);

    if (iter == interop_maps[device].end())
        interop_maps[device][key] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, plot->vertices(), NULL);

    return interop_maps[device][key];
}

cl::Buffer* InteropManager::getBufferResource(const forge::Histogram* hist)
{
    void * key = (void*)hist;
    int device = getActiveDeviceId();
    iter_t iter = interop_maps[device].find(key);

    if (iter == interop_maps[device].end())
        interop_maps[device][key] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, hist->vertices(), NULL);

    return interop_maps[device][key];
}

cl::Buffer* InteropManager::getBufferResource(const forge::Surface* surface)
{
    void * key = (void*)surface;
    int device = getActiveDeviceId();
    iter_t iter = interop_maps[device].find(key);

    if (iter == interop_maps[device].end())
        interop_maps[device][key] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, surface->vertices(), NULL);

    return interop_maps[device][key];
}

}

#endif

