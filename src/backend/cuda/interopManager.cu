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

#include <interopManager.hpp>
#include <err_cuda.hpp>
#include <cstdio>

namespace cuda
{

void InteropManager::destroyResources()
{
    int n = getActiveDeviceId();
    for(iter_t iter = interop_maps[n].begin(); iter != interop_maps[n].end(); iter++) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(iter->second));
    }
}

InteropManager::~InteropManager()
{
    try {
        for(int i = 0; i < getDeviceCount(); i++) {
            setDevice(i);
            destroyResources();
        }
    } catch (AfError &ex) {

        const char* perr = getenv("AF_PRINT_ERRORS");

        if(perr && perr[0] != '0') {
            fprintf(stderr, "%s\n", ex.what());
        }
    }
}

InteropManager& InteropManager::getInstance()
{
    static InteropManager my_instance;
    return my_instance;
}

cudaGraphicsResource* InteropManager::getBufferResource(const fg::Image* key)
{
    int device = getActiveDeviceId();
    void* key_value = (void*)key;

    if(interop_maps[device].find(key_value) == interop_maps[device].end()) {
        cudaGraphicsResource *cudaPBOResource;
        // Register PBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, key->pbo(), cudaGraphicsMapFlagsWriteDiscard));
        interop_maps[device][key_value] = cudaPBOResource;
    }

    return interop_maps[device][key_value];
}

cudaGraphicsResource* InteropManager::getBufferResource(const fg::Plot* key)
{
    int device = getActiveDeviceId();
    void* key_value = (void*)key;

    iter_t iter = interop_maps[device].find(key_value);

    if(interop_maps[device].find(key_value) == interop_maps[device].end()) {
        cudaGraphicsResource *cudaVBOResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, key->vbo(), cudaGraphicsMapFlagsWriteDiscard));
        interop_maps[device][key_value] = cudaVBOResource;
    }

    return interop_maps[device][key_value];
}

cudaGraphicsResource* InteropManager::getBufferResource(const fg::Histogram* key)
{
    int device = getActiveDeviceId();
    void* key_value = (void*)key;

    iter_t iter = interop_maps[device].find(key_value);

    if(interop_maps[device].find(key_value) == interop_maps[device].end()) {
        cudaGraphicsResource *cudaVBOResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, key->vbo(), cudaGraphicsMapFlagsWriteDiscard));
        interop_maps[device][key_value] = cudaVBOResource;
    }

    return interop_maps[device][key_value];
}

}

#endif
