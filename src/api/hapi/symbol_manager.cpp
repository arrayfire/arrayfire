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
static const char* LIB_AF_BKND_NAME[] = {"afcpu.dll", "afcuda.dll", "afopencl.dll"};
#define RTLD_LAZY 0
#else
static const char* LIB_AF_BKND_NAME[] = {"libafcpu.so", "libafcuda.so", "libafopencl.so"};
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
    : backendBitFlag(0x0000), activeHandle(NULL), defaultHandle(NULL)
{
    // AF_BACKEND_DEFAULT enum value is 1 + last valid compute
    // backend in af_backend enum, hence it represents the number
    // of valid backends in ArrayFire framework
    unsigned bkndFlag = 0x0001;
    for(int i=0; i<AF_BACKEND_DEFAULT; ++i) {
        bkndHandles[i] = openDynLibrary(LIB_AF_BKND_NAME[i]);
        if (bkndHandles[i]) {
            backendBitFlag |= bkndFlag;
            activeHandle = bkndHandles[i];
        }
        bkndFlag = bkndFlag << 1;
    }
    // Keep a copy of default order handle
    // inorder to use it in ::setBackend when
    // the user passes AF_BACKEND_DEFAULT
    defaultHandle = activeHandle;
}

AFSymbolManager::~AFSymbolManager()
{
    unsigned bkndFlag = 0x0001;
    for(int i=0; i<AF_BACKEND_DEFAULT; ++i) {
        if (bkndFlag & backendBitFlag)
            closeDynLibrary(bkndHandles[i]);
        bkndFlag = bkndFlag << 1;
    }
    backendBitFlag = 0x0000;
}

af_err AFSymbolManager::setBackend(af::Backend bknd)
{
    if (bknd==AF_BACKEND_DEFAULT) {
        activeHandle = defaultHandle;
        return AF_SUCCESS;
    }
    unsigned bkndFlag = 0x0001;
    if((bkndFlag << bknd) & backendBitFlag) {
        activeHandle = bkndHandles[bknd];
        return AF_SUCCESS;
    } else {
        return AF_ERR_LOAD_LIB;
    }
}
