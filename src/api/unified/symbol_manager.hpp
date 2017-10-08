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
#include <common/util.hpp>
#include <common/err_common.hpp>

#if defined(OS_WIN)
#include <Windows.h>
typedef HMODULE LibHandle;
#else
#include <dlfcn.h>
typedef void* LibHandle;
#endif

#include <array>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace unified
{

const int NUM_BACKENDS = 3;
const int NUM_ENV_VARS = 2;

#define UNIFIED_ERROR_LOAD_LIB()                                        \
    AF_RETURN_ERROR("Failed to load dynamic library. "                  \
                    "See http://www.arrayfire.com/docs/unifiedbackend.htm " \
                    "for instructions to set up environment for Unified backend.", \
                    AF_ERR_LOAD_LIB)

static inline int backend_index(af::Backend be) {
    switch (be) {
        case AF_BACKEND_CPU: return 0;
        case AF_BACKEND_CUDA: return 1;
        case AF_BACKEND_OPENCL: return 2;
        default: return -1;
    }
}

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
            typedef af_err(*af_func)(CalleeArgs...);
            if (!activeHandle) {
                UNIFIED_ERROR_LOAD_LIB();
            }
            thread_local std::array<std::unordered_map<const char*, af_func>, NUM_BACKENDS> funcHandles;

            int index = backend_index(getActiveBackend());
            af_func& funcHandle = funcHandles[index][symbolName];

            if (!funcHandle) {
#if defined(OS_WIN)
                funcHandle = (af_func)GetProcAddress(activeHandle, symbolName);
#else
                funcHandle = (af_func)dlsym(activeHandle, symbolName);
#endif
            }
            if (!funcHandle) {
                std::string str = "Failed to load symbol: ";
                str += symbolName;
                AF_RETURN_ERROR(str.c_str(),
                                AF_ERR_LOAD_SYM);
            }

            return funcHandle(args...);
        }

        LibHandle getHandle() { return activeHandle; }

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
#define CHECK_ARRAYS(...) do {                                          \
        af_backend backendId = unified::AFSymbolManager::getInstance().getActiveBackend(); \
        if(!unified::checkArrays(backendId, __VA_ARGS__))               \
            AF_RETURN_ERROR("Input array does not belong to current backend", \
                            AF_ERR_ARR_BKND_MISMATCH);                  \
    } while(0)

#if defined(OS_WIN)
#define CALL(...) unified::AFSymbolManager::getInstance().call(__FUNCTION__, __VA_ARGS__)
#define CALL_NO_PARAMS() unified::AFSymbolManager::getInstance().call(__FUNCTION__)
#else
#define CALL(...) unified::AFSymbolManager::getInstance().call(__func__, __VA_ARGS__)
#define CALL_NO_PARAMS() unified::AFSymbolManager::getInstance().call(__func__)
#endif

#if defined(OS_WIN)
#define LOAD_SYMBOL() GetProcAddress(unified::AFSymbolManager::getInstance().getHandle(), __FUNCTION__)
#else
#define LOAD_SYMBOL() dlsym(unified::AFSymbolManager::getInstance().getHandle(), __func__)
#endif
