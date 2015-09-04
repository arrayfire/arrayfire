/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "symbol_manager.hpp"

#if defined(OS_WIN)
static const char* LIB_AF_BKND_NAME[NUM_BACKENDS] = {"afcpu.dll", "afcuda.dll", "afopencl.dll"};
#define RTLD_LAZY 0
#else
static const char* LIB_AF_BKND_NAME[NUM_BACKENDS] = {"libafcpu.so", "libafcuda.so", "libafopencl.so"};
#endif

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
    : activeHandle(NULL), defaultHandle(NULL), numBackends(0)
{

    for(int i=0; i<NUM_BACKENDS; ++i) {
        bkndHandles[i] = openDynLibrary(LIB_AF_BKND_NAME[i]);
        if (bkndHandles[i]) {
            activeHandle = bkndHandles[i];
            numBackends++;
        }
    }
    // Keep a copy of default order handle
    // inorder to use it in ::setBackend when
    // the user passes AF_BACKEND_DEFAULT
    defaultHandle = activeHandle;
}

AFSymbolManager::~AFSymbolManager()
{
    for(int i=0; i<NUM_BACKENDS; ++i) {
        if (bkndHandles[i]) {
            closeDynLibrary(bkndHandles[i]);
        }
    }
}

unsigned AFSymbolManager::getBackendCount()
{
    return numBackends;
}

af_err AFSymbolManager::setBackend(af::Backend bknd)
{
    if (bknd==AF_BACKEND_DEFAULT) {
        if (defaultHandle) {
            activeHandle = defaultHandle;
            return AF_SUCCESS;
        } else
            return AF_ERR_LOAD_LIB;
    }
    unsigned idx = bknd - 1;
    if(bkndHandles[idx]) {
        activeHandle = bkndHandles[idx];
        return AF_SUCCESS;
    } else {
        return AF_ERR_LOAD_LIB;
    }
}
