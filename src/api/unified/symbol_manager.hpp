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
#else
#include <dlfcn.h>
typedef void* LibHandle;
#endif

const int NUM_BACKENDS = 3;
const int NUM_ENV_VARS = 2;

class AFSymbolManager {
    public:
        static AFSymbolManager& getInstance();

        ~AFSymbolManager();

        unsigned getBackendCount();

        int getAvailableBackends();

        af_err setBackend(af::Backend bnkd);

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
};

#if defined(OS_WIN)
#define CALL(...) AFSymbolManager::getInstance().call(__FUNCTION__, __VA_ARGS__)
#define CALL_NO_PARAMS() AFSymbolManager::getInstance().call(__FUNCTION__)
#else
#define CALL(...) AFSymbolManager::getInstance().call(__func__, __VA_ARGS__)
#define CALL_NO_PARAMS() AFSymbolManager::getInstance().call(__func__)
#endif
