/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "symbol_manager.hpp"
#include <af/version.h>
#include <common/module_loading.hpp>

#include <cmath>
#include <string>
#include <type_traits>


#ifndef WIN_OS
#include <dlfcn.h>
#else
#include <Windows.h>
#endif

using common::loadLibrary;
using common::unloadLibrary;

using std::extent;
using std::string;

namespace unified
{

#if defined(OS_WIN)
static const char* LIB_AF_BKND_PREFIX = "";
static const char* LIB_AF_BKND_SUFFIX = ".dll";
#define PATH_SEPARATOR "\\"
#define RTLD_LAZY 0
#else

#if defined(__APPLE__)
#  define SO_SUFFIX_HELPER(VER) "." #VER ".dylib"
#else
#  define SO_SUFFIX_HELPER(VER) ".so." #VER
#endif
   static const char* LIB_AF_BKND_PREFIX = "lib";
#  define PATH_SEPARATOR "/"

#  define GET_SO_SUFFIX(VER) SO_SUFFIX_HELPER(VER)
   static const char* LIB_AF_BKND_SUFFIX = GET_SO_SUFFIX(AF_VERSION_MAJOR);
#endif

string getBkndLibName(const af_backend backend) {
    string ret;
    switch (backend) {
        case AF_BACKEND_CUDA: ret = string(LIB_AF_BKND_PREFIX) + "afcuda" + LIB_AF_BKND_SUFFIX; break;
        case AF_BACKEND_OPENCL: ret = string(LIB_AF_BKND_PREFIX) + "afopencl" + LIB_AF_BKND_SUFFIX; break;
        case AF_BACKEND_CPU: ret = string(LIB_AF_BKND_PREFIX) + "afcpu" + LIB_AF_BKND_SUFFIX; break;
        default: assert(1!=1 && "Invalid backend");
    }
    return ret;
}
string getBackendDirectoryName(const af_backend backend) {
    string ret;
    switch (backend) {
        case AF_BACKEND_CUDA: ret = "cuda"; break;
        case AF_BACKEND_OPENCL: ret = "opencl"; break;
        case AF_BACKEND_CPU: ret = "cpu"; break;
        default: assert(1!=1 && "Invalid backend");
      }
    return ret;
}

string join_path(string first) {
    return first;
}

template<typename... ARGS>
string join_path(string first, ARGS... args) {
    if(first.empty()) { return join_path(args...); }
    else              { return first + PATH_SEPARATOR + join_path(args...); }
}

/*flag parameter is not used on windows platform */
LibHandle openDynLibrary(const af_backend bknd_idx, int flag=RTLD_LAZY)
{
    // The default search path is the colon separated list of paths stored in
    // the environment variables:
    string bkndLibName = getBkndLibName(bknd_idx);
    string show_flag = getEnvVar("AF_SHOW_LOAD_PATH");
    bool show_load_path = show_flag=="1";

    string paths[] = {
        "",  // Default paths
        ".", // Shared libraries in current directory
        // Running from the CMake Build directory
        join_path(".", "src", "backend", getBackendDirectoryName(bknd_idx)),
        // Running from the test directory
        join_path("..", "src", "backend", getBackendDirectoryName(bknd_idx)),
        // Environment variable PATHS
        join_path(getEnvVar("AF_BUILD_PATH"), "src", "backend", getBackendDirectoryName(bknd_idx)),
        join_path(getEnvVar("AF_PATH"), "lib"),
        join_path(getEnvVar("AF_PATH"), "lib64"),

        // Common install paths
#if !defined(OS_WIN)
        "/opt/arrayfire-3/lib/",
        "/opt/arrayfire/lib/",
        "/usr/local/lib/",
        "/usr/local/arrayfire/lib/"
#else
        join_path(getEnvVar("ProgramFiles"), "ArrayFire", "lib"),
        join_path(getEnvVar("ProgramFiles"), "ArrayFire", "v3", "lib")
#endif
    };

    LibHandle retVal = nullptr;
    for (int i = 0; i < extent<decltype(paths)>::value; i++) {
        if (retVal = common::loadLibrary(join_path(paths[i], bkndLibName).c_str())) {
            if (show_load_path) {
                printf("Using %s\n", bkndLibName.c_str());
            }
            break;
        }
    }

    return retVal;
}

void closeDynLibrary(LibHandle handle)
{
    unloadLibrary(handle);
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
    static const af_backend order[] = { AF_BACKEND_CUDA,
                                        AF_BACKEND_OPENCL,
                                        AF_BACKEND_CPU};

    // Decremeting loop. The last successful backend loaded will be the most prefered one.
    for(int i = NUM_BACKENDS - 1; i >= 0; i--) {
        int backend = order[i] >> 1; // 2 4 1 -> 1 2 0
        bkndHandles[backend] = openDynLibrary(order[i]);
        if (bkndHandles[backend]) {
            activeHandle = bkndHandles[backend];
            activeBackend = (af_backend)order[i];
            numBackends++;
            backendsAvailable += order[i];
        }
    }

    // Keep a copy of default order handle inorder to use it in ::setBackend
    // when the user passes AF_BACKEND_DEFAULT
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
