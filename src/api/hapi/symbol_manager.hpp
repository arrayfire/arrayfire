/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <string>
#include <stdlib.h>
#if defined(OS_WIN)
#include <Windows.h>
typedef HMODULE LibHandle;
#define RTLD_LAZY 0
#define LIB_AF_CPU_NAME "afcpu.dll"
#define LIB_AF_CUDA_NAME "afcuda.dll"
#define LIB_AF_OCL_NAME "afopencl.dll"
#else
#include <dlfcn.h>
typedef void* LibHandle;
#define LIB_AF_CPU_NAME "libafcpu.so"
#define LIB_AF_CUDA_NAME "libafcuda.so"
#define LIB_AF_OCL_NAME "libafopencl.so"
#endif

class AFSymbolManager {
    public:
        static AFSymbolManager& getInstance();

        ~AFSymbolManager();

        af_err setBackend(af::Backend bnkd);

        template<typename... CalleeArgs>
        af_err call(const char* symbolName, CalleeArgs... args) {
            typedef af_err(*af_func)(CalleeArgs...);
            af_func funcHandle;
#if defined(OS_WIN)
            funcHandle = (af_func)GetProcAddress(activeHandle, symbolName);
#else
            funcHandle = (af_func)dlsym(activeHandle, symbolName);
#endif
            if (!funcHandle) {
                return AF_ERR_SYM_LOAD;
            }
            return funcHandle(args...);
        }

    protected:
        AFSymbolManager();

        // Following two declarations are required to
        // avoid copying accidental copy/assignment
        // of instance returned by getInstance to other
        // variables
        AFSymbolManager(AFSymbolManager const&);
        void operator=(AFSymbolManager const&);

    private:
        bool isCPULoaded;
        bool isCUDALoaded;
        bool isOCLLoaded;

        LibHandle cpuHandle;
        LibHandle cudaHandle;
        LibHandle oclHandle;

        af::Backend activeBknd;
        LibHandle activeHandle;
};

#if defined(OS_WIN)
#define CALL(...) AFSymbolManager::getInstance().call(__FUNCTION__, __VA_ARGS__)
#else
#define CALL(...) AFSymbolManager::getInstance().call(__func__, __VA_ARGS__)
#endif
