/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "symbol_manager.hpp"
#include <algorithm>
#include <string>
#include <cmath>

using std::string;
using std::replace;

namespace unified
{

static const string LIB_AF_BKND_NAME[NUM_BACKENDS] = {"cpu", "cuda", "opencl"};
#if defined(OS_WIN)
static const string LIB_AF_BKND_PREFIX = "af";
static const string LIB_AF_BKND_SUFFIX = ".dll";
#define RTLD_LAZY 0
#else
static const string LIB_AF_BKND_PREFIX = "libaf";
#if defined(__APPLE__)
static const string LIB_AF_BKND_SUFFIX = ".dylib";
#else
static const string LIB_AF_BKND_SUFFIX = ".so";
#endif // APPLE
#endif

static const string LIB_AF_ENVARS[NUM_ENV_VARS] = {"AF_PATH", "AF_BUILD_PATH"};
static const string LIB_AF_RPATHS[NUM_ENV_VARS] = {"/lib/", "/src/backend/"};
static const bool LIB_AF_RPATH_SUFFIX[NUM_ENV_VARS] = {false, true};

inline string getBkndLibName(const int backend_index)
{
    int i = backend_index >=0 && backend_index<NUM_BACKENDS ? backend_index : 0;
    return LIB_AF_BKND_PREFIX + LIB_AF_BKND_NAME[i] + LIB_AF_BKND_SUFFIX;
}

inline std::string getEnvVar(const std::string &key)
{
#if defined(OS_WIN)
    DWORD bufSize = 32767; // limit according to GetEnvironment Variable documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetEnvironmentVariable(key.c_str(), &retVal[0], bufSize);
    if (!bufSize) {
        return string("");
    } else {
        retVal.resize(bufSize);
        return retVal;
    }
#else
    char * str = getenv(key.c_str());
    return str==NULL ? string("") : string(str);
#endif
}

/*flag parameter is not used on windows platform */
LibHandle openDynLibrary(const int bknd_idx, int flag=RTLD_LAZY)
{
    /*
     * The default search path is the colon separated list of
     * paths stored in the environment variables:
     * * LD_LIBRARY_PATH(Linux/Unix/Apple)
     * * DYLD_LIBRARY_PATH (Apple)
     * * PATH (Windows)
    */
    string bkndName = getBkndLibName(bknd_idx);
    string show_flag = getEnvVar("AF_SHOW_LOAD_PATH");
    bool show_load_path = show_flag=="1";

#if defined(OS_WIN)
    HMODULE retVal = LoadLibrary(bkndName.c_str());
#else
    LibHandle retVal = dlopen(bkndName.c_str(), flag);
#endif
    if(retVal != NULL) { // Success
        if (show_load_path)
            printf("Using %s from system path\n", bkndName.c_str());
    } else {
        /*
         * In the event that dlopen returns NULL, search for the lib
         * in hard coded paths based on the environment variables
         * defined in the constant string array LIB_AF_PATHS
         * * AF_PATH
         * * AF_BUILD_PATH
         *
         * Note: This does not guarantee successful loading as the dependent
         * libraries may still not load
        */
        for (int i=0; i<NUM_ENV_VARS; ++i) {
            string abs_path = getEnvVar(LIB_AF_ENVARS[i])
                                 + LIB_AF_RPATHS[i]
                                 + (LIB_AF_RPATH_SUFFIX[i] ? LIB_AF_BKND_NAME[bknd_idx]+"/" : "")
                                 + bkndName;
#if defined(OS_WIN)
            replace(abs_path.begin(), abs_path.end(), '/', '\\');
            retVal = LoadLibrary(abs_path.c_str());
#else
            retVal = dlopen(abs_path.c_str(), flag);
#endif
            if (retVal!=NULL) {
                if (show_load_path)
                    printf("Using %s\n", abs_path.c_str());
                // if the current absolute path based dlopen
                // search is a success, then abandon search
                // and proceed for compute
                break;
            }
        }
    }
    return retVal;
}

void closeDynLibrary(LibHandle handle)
{
#if defined(OS_WIN)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

AFSymbolManager& AFSymbolManager::getInstance()
{
    static AFSymbolManager symbolManager;
    return symbolManager;
}

AFSymbolManager::AFSymbolManager()
    : activeHandle(NULL), defaultHandle(NULL), numBackends(0), backendsAvailable(0)
{
    // In order of priority.
    static const int order[] = {AF_BACKEND_CUDA,        // 1 -> Most Preferred
                                AF_BACKEND_OPENCL,      // 4 -> Preferred if CUDA unavailable
                                AF_BACKEND_CPU};        // 2 -> Preferred if CUDA and OpenCL unavailable

    // Decremeting loop. The last successful backend loaded will be the most prefered one.
    for(int i = NUM_BACKENDS - 1; i >= 0; i--) {
        int backend = order[i] >> 1;    // Convert order[1, 4, 2] -> backend[0, 2, 1]
        bkndHandles[backend] = openDynLibrary(backend);
        if (bkndHandles[backend]) {
            activeHandle = bkndHandles[backend];
            activeBackend = (af_backend)order[i];
            numBackends++;
            backendsAvailable += order[i];
        }
    }
    // Keep a copy of default order handle
    // inorder to use it in ::setBackend when
    // the user passes AF_BACKEND_DEFAULT
    defaultHandle = activeHandle;
    defaultBackend = activeBackend;
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

int AFSymbolManager::getAvailableBackends()
{
    return backendsAvailable;
}

af_err AFSymbolManager::setBackend(af::Backend bknd)
{
    if (bknd==AF_BACKEND_DEFAULT) {
        if (defaultHandle) {
            activeHandle = defaultHandle;
            activeBackend = defaultBackend;
            return AF_SUCCESS;
        } else
            return AF_ERR_LOAD_LIB;
    }
    int idx = bknd >> 1;    // Convert 1, 2, 4 -> 0, 1, 2
    if(bkndHandles[idx]) {
        activeHandle = bkndHandles[idx];
        activeBackend = bknd;
        return AF_SUCCESS;
    } else {
        return AF_ERR_LOAD_LIB;
    }
}

bool checkArray(af_backend activeBackend, af_array a)
{
    // Convert af_array into int to retrieve the backend info.
    // See ArrayInfo.hpp for more
    af_backend backend = (af_backend)0;
    unified::AFSymbolManager::getInstance().call("af_get_backend_id", &backend, a);
    return backend == activeBackend;
}

bool checkArrays(af_backend activeBackend)
{
    // Dummy
    return true;
}

} // namespace unified
