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
    typedef std::vector<CGR_t>::iterator CGRIter_t;

    int n = getActiveDeviceId();
    for(iter_t iter = interop_maps[n].begin(); iter != interop_maps[n].end(); iter++) {
        for(CGRIter_t ct = (iter->second).begin(); ct != (iter->second).end(); ct++) {
            CUDA_CHECK(cudaGraphicsUnregisterResource(*ct));
        }
        (iter->second).clear();
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

interop_t& InteropManager::getDeviceMap(int device)
{
    return (device == -1) ? interop_maps[getActiveDeviceId()] : interop_maps[device];
}

CGR_t* InteropManager::getBufferResource(const forge::Image* key)
{
    void* key_value = (void*)key;
    interop_t& i_map = getDeviceMap();

    if(i_map.find(key_value) == i_map.end()) {
        CGR_t pboResource;
        // Register PBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pboResource, key->pbo(), cudaGraphicsMapFlagsWriteDiscard));
        // TODO:
        // A way to store multiple buffers and take PBO/CBO etc as
        // argument and return the appropriate buffer
        std::vector<CGR_t> vec(1);
        vec[0] = pboResource;
        i_map[key_value] = vec;
    }

    return &i_map[key_value].front();
}

CGR_t* InteropManager::getBufferResource(const forge::Plot* key)
{
    void* key_value = (void*)key;
    interop_t& i_map = getDeviceMap();

    iter_t iter = i_map.find(key_value);

    if(iter == i_map.end()) {
        CGR_t vboResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vboResource, key->vertices(), cudaGraphicsMapFlagsWriteDiscard));
        // TODO:
        // A way to store multiple buffers and take PBO/CBO etc as
        // argument and return the appropriate buffer
        std::vector<CGR_t> vec(1);
        vec[0] = vboResource;
        i_map[key_value] = vec;
    }

    return &i_map[key_value].front();
}

CGR_t* InteropManager::getBufferResource(const forge::Histogram* key)
{
    void* key_value = (void*)key;
    interop_t& i_map = getDeviceMap();

    iter_t iter = i_map.find(key_value);

    if(iter == i_map.end()) {
        CGR_t vboResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vboResource, key->vertices(), cudaGraphicsMapFlagsWriteDiscard));
        // TODO:
        // A way to store multiple buffers and take PBO/CBO etc as
        // argument and return the appropriate buffer
        std::vector<CGR_t> vec(1);
        vec[0] = vboResource;
        i_map[key_value] = vec;
    }

    return &i_map[key_value].front();
}

CGR_t* InteropManager::getBufferResource(const forge::Surface* key)
{
    void* key_value = (void*)key;
    interop_t& i_map = getDeviceMap();

    iter_t iter = i_map.find(key_value);

    if(iter == i_map.end()) {
        CGR_t  vboResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vboResource, key->vertices(), cudaGraphicsMapFlagsWriteDiscard));
        // TODO:
        // A way to store multiple buffers and take PBO/CBO etc as
        // argument and return the appropriate buffer
        std::vector<CGR_t> vec(1);
        vec[0] = vboResource;
        i_map[key_value] = vec;
    }

    return &i_map[key_value].front();
}

CGR_t* InteropManager::getBufferResource(const forge::VectorField* key)
{
    void* key_value = (void*)key;
    interop_t& i_map = getDeviceMap();

    iter_t iter = i_map.find(key_value);

    if(iter == i_map.end()) {
        CGR_t   pResource;
        CGR_t   dResource;
        // Register VBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pResource, key->vertices(), cudaGraphicsMapFlagsWriteDiscard));
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&dResource, key->directions(), cudaGraphicsMapFlagsWriteDiscard));
        // TODO:
        // A way to store multiple buffers and take PBO/CBO etc as
        // argument and return the appropriate buffer
        std::vector<CGR_t> vec(2);
        vec[0] = pResource;
        vec[1] = dResource;
        i_map[key_value] = vec;
    }

    return &i_map[key_value].front();
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
