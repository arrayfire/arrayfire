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

/*flag parameter is not used on windows platform */
LibHandle openDynLibrary(const char* dlName, int flag=RTLD_LAZY)
{
#if defined(OS_WIN)
    HMODULE retVal = LoadLibrary(dlName);
    if (retVal == NULL) {
        retVal = LoadLibraryEx(dlName, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }
    return retVal;
#else
    return dlopen(dlName, flag);
#endif
}

void closeDynLibrary(LibHandle handle)
{
#if defined(OS_WIN)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

AFSymbolManager::AFSymbolManager()
    : isCPULoaded(false), isCUDALoaded(false), isOCLLoaded(false)
{
    cpuHandle = openDynLibrary(LIB_AF_CPU_NAME);
    if (cpuHandle) {
        isCPULoaded = true;
        activeHandle = cpuHandle;
    }
    cudaHandle = openDynLibrary(LIB_AF_CUDA_NAME);
    if (cudaHandle) {
        isCUDALoaded = true;
        activeHandle = cudaHandle;
    }
    oclHandle = openDynLibrary(LIB_AF_OCL_NAME);
    if (oclHandle) {
        isOCLLoaded = true;
        activeHandle = oclHandle;
    }
}

AFSymbolManager::~AFSymbolManager()
{
    if (isCPULoaded) {
        closeDynLibrary(cpuHandle);
        isCPULoaded = false;
    }
    if (isCUDALoaded) {
        closeDynLibrary(cudaHandle);
        isCUDALoaded = false;
    }
    if (isOCLLoaded) {
        closeDynLibrary(oclHandle);
        isOCLLoaded = false;
    }
}

af_err AFSymbolManager::setBackend(af::Backend bknd)
{
    af_err retCode = AF_SUCCESS;
    activeBknd = bknd;
    switch (activeBknd) {
        case af::Backend::AF_BACKEND_CPU:
            if(isCPULoaded)
                activeHandle = cpuHandle;
            else
                retCode = AF_ERR_LOAD_LIB;
            break;
        case af::Backend::AF_BACKEND_CUDA:
            if(isCUDALoaded)
                activeHandle = cudaHandle;
            else
                retCode = AF_ERR_LOAD_LIB;
            break;
        case af::Backend::AF_BACKEND_OPENCL:
            if (isOCLLoaded)
                activeHandle = oclHandle;
            else
                retCode = AF_ERR_LOAD_LIB;
            break;
    }
    return retCode;
}
