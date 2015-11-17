/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/defines.h>
#include <string>
#include <stdlib.h>
#if defined(OS_WIN)
#include <Windows.h>
typedef HMODULE LibHandle;
#else
#include <dlfcn.h>
typedef void* LibHandle;
#endif

namespace unified
{

const int NUM_BACKENDS = 3;
const int NUM_ENV_VARS = 2;

class AFSymbolManager {
    public:
        static AFSymbolManager& getInstance();

        ~AFSymbolManager();

        unsigned getBackendCount();

        int getAvailableBackends();

        af_err setBackend(af::Backend bnkd);

        af::Backend getActiveBackend() { return activeBackend; }

        template<typename... CalleeArgs>
        af_err call(const char* symbolName, CalleeArgs... args) {
            if (!activeHandle)
                return AF_ERR_LOAD_LIB;
            typedef af_err(*af_func)(CalleeArgs...);
            af_func funcHandle;
#if defined(OS_WIN)
            funcHandle = (af_func)GetProcAddress(activeHandle, symbolName);
#else
            funcHandle = (af_func)dlsym(activeHandle, symbolName);
#endif
            if (!funcHandle) {
                return AF_ERR_LOAD_SYM;
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

        LibHandle bkndHandles[NUM_BACKENDS];

        LibHandle activeHandle;
        LibHandle defaultHandle;
        unsigned numBackends;
        int backendsAvailable;
        af_backend activeBackend;
        af_backend defaultBackend;
};

// Helper functions to ensure all the input arrays are on the active backend
bool checkArray(af_backend activeBackend, af_array a);
bool checkArrays(af_backend activeBackend);

template<typename T, typename... Args>
bool checkArrays(af_backend activeBackend, T a, Args... arg)
{
    return checkArray(activeBackend, a) && checkArrays(activeBackend, arg...);
}

} // namespace unified

// Macro to check af_array as inputs. The arguments to this macro should be
// only input af_arrays. Not outputs or other types.
#define CHECK_ARRAYS(...) do {                                                              \
    af_backend backendId = unified::AFSymbolManager::getInstance().getActiveBackend();      \
    if(!unified::checkArrays(backendId, __VA_ARGS__))                                       \
        return AF_ERR_ARR_BKND_MISMATCH;                                                    \
} while(0);

#if defined(OS_WIN)
#define CALL(...) unified::AFSymbolManager::getInstance().call(__FUNCTION__, __VA_ARGS__)
#define CALL_NO_PARAMS() unified::AFSymbolManager::getInstance().call(__FUNCTION__)
#else
#define CALL(...) unified::AFSymbolManager::getInstance().call(__func__, __VA_ARGS__)
#define CALL_NO_PARAMS() unified::AFSymbolManager::getInstance().call(__func__)
#endif
