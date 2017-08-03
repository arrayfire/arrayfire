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
#include <vector>
#include <string>
#include <cmath>
#include <af/version.h>

using std::string;
using std::replace;

namespace unified
{

static const char* LIB_AF_BKND_NAME[NUM_BACKENDS] = {"cpu", "cuda", "opencl"};
#if defined(OS_WIN)
static const char* LIB_AF_BKND_PREFIX = "af";
static const char* LIB_AF_BKND_SUFFIX = ".dll";
#define RTLD_LAZY 0
#else
#if defined(__APPLE__)
#define SO_SUFFIX_HELPER(VER) "." #VER ".dylib"
#else
#define SO_SUFFIX_HELPER(VER) ".so." #VER
#endif // APPLE
static const char* LIB_AF_BKND_PREFIX = "libaf";

#define GET_SO_SUFFIX(VER) SO_SUFFIX_HELPER(VER)
static const char* LIB_AF_BKND_SUFFIX = GET_SO_SUFFIX(AF_VERSION_MAJOR);
#endif

static const char* LIB_AF_ENVARS[NUM_ENV_VARS] = {"AF_PATH", "AF_BUILD_PATH"};
static const char* LIB_AF_RPATHS[NUM_ENV_VARS] = {"/lib/", "/src/backend/"};
static const bool LIB_AF_RPATH_SUFFIX[NUM_ENV_VARS] = {false, true};

inline string getBkndLibName(const int backend_index)
{
    int i = backend_index >=0 && backend_index<NUM_BACKENDS ? backend_index : 0;
    return string(LIB_AF_BKND_PREFIX) + LIB_AF_BKND_NAME[i] + LIB_AF_BKND_SUFFIX;
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
    string bkndLibName = getBkndLibName(bknd_idx);
    string show_flag = getEnvVar("AF_SHOW_LOAD_PATH");
    bool show_load_path = show_flag=="1";

#if defined(OS_WIN)
    HMODULE retVal = LoadLibrary(bkndLibName.c_str());
#else
    LibHandle retVal = dlopen(bkndLibName.c_str(), flag);
#endif
    if(retVal != NULL) { // Success
        if (show_load_path)
            printf("Using %s from system path\n", bkndLibName.c_str());
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
                                 + (LIB_AF_RPATH_SUFFIX[i] ? LIB_AF_BKND_NAME[bknd_idx]+ string("/") : "")
                                 + bkndLibName;
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

#if !defined(OS_WIN)
        /*
         * If Linux/OSX, then the following are also checked
         * (only if lib is not found)
         * /opt/arrayfire/lib
         * /opt/arrayfire-3/lib
         * /usr/local/lib
         * /usr/local/arrayfire/lib
         * /usr/local/arrayfire-3/lib
        */
        if (retVal == NULL) {
            static const
            std::vector<std::string> extraLibPaths {"/opt/arrayfire-3/lib/",
                                                    "/opt/arrayfire/lib/",
                                                    "/usr/local/lib/",
                                                    "/usr/local/arrayfire-3/lib/",
                                                    "/usr/local/arrayfire/lib/",
                                                   };

            for (auto libPath: extraLibPaths) {
                string abs_path = libPath + bkndLibName;
                retVal = dlopen(abs_path.c_str(), flag);
                if (retVal != NULL) {
                    if (show_load_path)
                        printf("Using %s\n", abs_path.c_str());
                    // if the current absolute path based dlopen
                    // search is a success, then abandon search
                    // and proceed for compute
                    break;
                }
            }
        }
#endif
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
    thread_local AFSymbolManager symbolManager;
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
        } else {
            UNIFIED_ERROR_LOAD_LIB();
        }
    }
    int idx = bknd >> 1;    // Convert 1, 2, 4 -> 0, 1, 2
    if(bkndHandles[idx]) {
        activeHandle = bkndHandles[idx];
        activeBackend = bknd;
        return AF_SUCCESS;
    } else {
        UNIFIED_ERROR_LOAD_LIB();
    }
}

bool checkArray(af_backend activeBackend, af_array a)
{
    // Convert af_array into int to retrieve the backend info.
    // See ArrayInfo.hpp for more
    af_backend backend = (af_backend)0;

    // This condition is required so that the invalid args tests for unified
    // backend return the expected error rather than AF_ERR_ARR_BKND_MISMATCH
    // Since a = 0, does not have a backend specified, it should be a
    // AF_ERR_ARG instead of AF_ERR_ARR_BKND_MISMATCH
    if(a == 0) return true;

    unified::AFSymbolManager::getInstance().call("af_get_backend_id", &backend, a);
    return backend == activeBackend;
}

bool checkArrays(af_backend activeBackend)
{
    // Dummy
    return true;
}

} // namespace unified
