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
    typedef std::vector<cl::Buffer *>::iterator buffer_t;
    int n = getActiveDeviceId();
    for(iter_t iter = interop_maps[n].begin(); iter != interop_maps[n].end(); iter++) {
        for(buffer_t bt = (iter->second).begin(); bt != (iter->second).end(); bt++) {
            delete *bt;
        }
        (iter->second).clear();
    }
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

interop_t& InteropManager::getDeviceMap(int device)
{
    return (device == -1) ? interop_maps[getActiveDeviceId()] : interop_maps[device];
}

cl::Buffer** InteropManager::getBufferResource(const forge::Image* image)
{
    void * key = (void*)image;
    interop_t& i_map = getDeviceMap();
    iter_t iter = i_map.find(key);

    if (iter == i_map.end()) {
        std::vector<cl::Buffer *> vec(1);
        vec[0] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, image->pixels(), NULL);
        i_map[key] = vec;
    }

    return &i_map[key].front();
}

cl::Buffer** InteropManager::getBufferResource(const forge::Plot* plot)
{
    void * key = (void*)plot;
    interop_t& i_map = getDeviceMap();
    iter_t iter = i_map.find(key);

    if (iter == i_map.end()) {
        std::vector<cl::Buffer *> vec(1);
        vec[0] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, plot->vertices(), NULL);
        i_map[key] = vec;
    }

    return &i_map[key].front();
}

cl::Buffer** InteropManager::getBufferResource(const forge::Histogram* hist)
{
    void * key = (void*)hist;
    interop_t& i_map = getDeviceMap();
    iter_t iter = i_map.find(key);

    if (iter == i_map.end()) {
        std::vector<cl::Buffer *> vec(1);
        vec[0] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, hist->vertices(), NULL);
        i_map[key] = vec;
    }

    return &i_map[key].front();
}

cl::Buffer** InteropManager::getBufferResource(const forge::Surface* surface)
{
    void * key = (void*)surface;
    interop_t& i_map = getDeviceMap();
    iter_t iter = i_map.find(key);

    if (iter == i_map.end()) {
        std::vector<cl::Buffer *> vec(1);
        vec[0] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, surface->vertices(), NULL);
        i_map[key] = vec;
    }

    return &i_map[key].front();
}

cl::Buffer** InteropManager::getBufferResource(const forge::VectorField* vector_field)
{
    void * key = (void*)vector_field;
    interop_t& i_map = getDeviceMap();
    iter_t iter = i_map.find(key);

    if (iter == i_map.end()) {
        std::vector<cl::Buffer *> vec(2);
        vec[0] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, vector_field->vertices(), NULL);
        vec[1] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, vector_field->directions(), NULL);
        i_map[key] = vec;
    }

    return &i_map[key].front();
}

}

#endif

