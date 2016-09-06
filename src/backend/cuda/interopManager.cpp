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
#include <util.hpp>
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

        std::string perr = getEnvVar("AF_PRINT_ERRORS");
        if(!perr.empty()) {
            if(perr != "0")
                fprintf(stderr, "%s\n", ex.what());
        }
    }
}

InteropManager& InteropManager::getInstance()
{
    static InteropManager my_instance;
    return my_instance;
}

cudaGraphicsResource* InteropManager::getBufferResource(const forge::Image* key)
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

cudaGraphicsResource* InteropManager::getBufferResource(const forge::Plot* key)
{
    int device = getActiveDeviceId();
    void* key_value = (void*)key;

    iter_t iter = interop_maps[device].find(key_value);

    if(iter == interop_maps[device].end()) {
        cudaGraphicsResource *cudaVBOResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, key->vertices(), cudaGraphicsMapFlagsWriteDiscard));
        interop_maps[device][key_value] = cudaVBOResource;
    }

    return interop_maps[device][key_value];
}

cudaGraphicsResource* InteropManager::getBufferResource(const forge::Histogram* key)
{
    int device = getActiveDeviceId();
    void* key_value = (void*)key;

    iter_t iter = interop_maps[device].find(key_value);

    if(iter == interop_maps[device].end()) {
        cudaGraphicsResource *cudaVBOResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, key->vertices(), cudaGraphicsMapFlagsWriteDiscard));
        interop_maps[device][key_value] = cudaVBOResource;
    }

    return interop_maps[device][key_value];
}

cudaGraphicsResource* InteropManager::getBufferResource(const forge::Surface* key)
{
    int device = getActiveDeviceId();
    void* key_value = (void*)key;

    iter_t iter = interop_maps[device].find(key_value);

    if(iter == interop_maps[device].end()) {
        cudaGraphicsResource *cudaVBOResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, key->vertices(), cudaGraphicsMapFlagsWriteDiscard));
        interop_maps[device][key_value] = cudaVBOResource;
    }

    return interop_maps[device][key_value];
}

bool InteropManager::checkGraphicsInteropCapability()
{
    static bool run_once = true;
    static bool capable  = true;

    if(run_once) {
        unsigned int pCudaEnabledDeviceCount = 0;
        int pCudaGraphicsEnabledDeviceIds = 0;
        cudaGetLastError(); // Reset Errors
        cudaError_t err = cudaGLGetDevices(&pCudaEnabledDeviceCount, &pCudaGraphicsEnabledDeviceIds, getDeviceCount(), cudaGLDeviceListAll);
        if(err == 63) { // OS Support Failure - Happens when devices are only Tesla
            capable = false;
            printf("Warning: No CUDA Device capable of CUDA-OpenGL. CUDA-OpenGL Interop will use CPU fallback.\n");
            printf("Corresponding CUDA Error (%d): %s.\n", err, cudaGetErrorString(err));
            printf("This may happen if all CUDA Devices are in TCC Mode and/or not connected to a display.\n");
        }
        cudaGetLastError(); // Reset Errors
        run_once = false;
    }
    return capable;
}

}

#endif
