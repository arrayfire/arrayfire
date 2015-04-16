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

cl::Buffer* InteropManager::getBufferResource(const fg_image_handle image)
{
    int device = getActiveDeviceId();
    iter_t iter = interop_maps[device].find(image);

    if (iter == interop_maps[device].end())
        interop_maps[device][image] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, image->gl_PBO, NULL);

    return interop_maps[device][image];
}

cl::Buffer* InteropManager::getBufferResource(const fg_plot_handle plot)
{
    int device = getActiveDeviceId();
    iter_t iter = interop_maps[device].find(plot);

    if (iter == interop_maps[device].end()) {
        interop_maps[device][plot] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, plot->gl_vbo[0], NULL);
    } else {
        cl::Buffer *buf = iter->second;
        size_t bytes = buf->getInfo<CL_MEM_SIZE>();
        if(bytes != plot->vbosize) {
            delete iter->second;
            interop_maps[device][plot] = new cl::BufferGL(getContext(), CL_MEM_WRITE_ONLY, plot->gl_vbo[0], NULL);
        }
    }

    return interop_maps[device][plot];
}

}

#endif

