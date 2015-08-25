/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "symbol_manager.hpp"

AFSymbolManager& AFSymbolManager::getInstance()
{
    static AFSymbolManager symbolManager;
    return symbolManager;
}

AFSymbolManager::AFSymbolManager()
    : isCPULoaded(false), isCUDALoaded(false), isOCLLoaded(false)
{
    cpuHandle = dlopen("libafcpu.so", RTLD_LAZY);
    if (cpuHandle) {
        isCPULoaded = true;
        activeHandle = cpuHandle;
    }
    cudaHandle = dlopen("libafcuda.so", RTLD_LAZY);
    if (cudaHandle) {
        isCUDALoaded = true;
        activeHandle = cudaHandle;
    }
    oclHandle = dlopen("libafopencl.so", RTLD_LAZY);
    if (oclHandle) {
        isOCLLoaded = true;
        activeHandle = oclHandle;
    }
}

AFSymbolManager::~AFSymbolManager()
{
    if (isCPULoaded) {
        dlclose(cpuHandle);
        isCPULoaded = false;
    }
    if (isCUDALoaded) {
        dlclose(cudaHandle);
        isCUDALoaded = false;
    }
    if (isOCLLoaded) {
        dlclose(oclHandle);
        isOCLLoaded = false;
    }
}

void AFSymbolManager::setBackend(af::Backend bknd)
{
    activeBknd = bknd;
    switch (activeBknd) {
        case af::Backend::AF_BACKEND_CPU:
            if(isCPULoaded)
                activeHandle = cpuHandle;
            else
                throw std::logic_error("can't load afcpu library");
            break;
        case af::Backend::AF_BACKEND_CUDA:
            if(isCUDALoaded)
                activeHandle = cudaHandle;
            else
                throw std::logic_error("can't load afcuda library");
            break;
        case af::Backend::AF_BACKEND_OPENCL:
            if(isOCLLoaded)
                activeHandle = oclHandle;
            else
                throw std::logic_error("can't load afopencl library");
            break;
    }
}
