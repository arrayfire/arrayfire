/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/Logger.hpp>
#include <common/err_common.hpp>
#include <common/module_loading.hpp>
#include <common/util.hpp>
#include <af/backend.h>
#include <af/defines.h>

#include <spdlog/spdlog.h>
#include <array>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace arrayfire {
namespace unified {

const int NUM_BACKENDS = 4;

#define UNIFIED_ERROR_LOAD_LIB()                                       \
    AF_RETURN_ERROR(                                                   \
        "Failed to load dynamic library. "                             \
        "See http://www.arrayfire.com/docs/unifiedbackend.htm "        \
        "for instructions to set up environment for Unified backend.", \
        AF_ERR_LOAD_LIB)

static inline int backend_index(af::Backend be) {
    switch (be) {
        case AF_BACKEND_CPU: return 0;
        case AF_BACKEND_CUDA: return 1;
        case AF_BACKEND_OPENCL: return 2;
        case AF_BACKEND_ONEAPI: return 3;
        default: return -1;
    }
}

class AFSymbolManager {
   public:
    static AFSymbolManager& getInstance() {
        static AFSymbolManager* symbolManager = new AFSymbolManager();
        return *symbolManager;
    }

    ~AFSymbolManager();

    unsigned getBackendCount() const;
    int getAvailableBackends() const;
    af::Backend getDefaultBackend() { return defaultBackend; }
    LibHandle getDefaultHandle() { return defaultHandle; }

    spdlog::logger* getLogger();
    LibHandle getHandle(int idx) { return bkndHandles[idx]; }

   protected:
    AFSymbolManager();

    // Following two declarations are required to
    // avoid copying accidental copy/assignment
    // of instance returned by getInstance to other
    // variables
    AFSymbolManager(AFSymbolManager const&);
    void operator=(AFSymbolManager const&);

   private:
    LibHandle bkndHandles[NUM_BACKENDS]{};

    LibHandle defaultHandle;
    unsigned numBackends;
    int backendsAvailable;
    af_backend defaultBackend;
    std::shared_ptr<spdlog::logger> logger;
};

af_err setBackend(af::Backend bknd);

af::Backend& getActiveBackend();

LibHandle& getActiveHandle();

namespace {
bool checkArray(af_backend activeBackend, const af_array a) {
    // Convert af_array into int to retrieve the backend info.
    // See ArrayInfo.hpp for more
    af_backend backend = (af_backend)0;

    // This condition is required so that the invalid args tests for unified
    // backend return the expected error rather than AF_ERR_ARR_BKND_MISMATCH
    // Since a = 0, does not have a backend specified, it should be a
    // AF_ERR_ARG instead of AF_ERR_ARR_BKND_MISMATCH
    if (a == 0) return true;

    af_get_backend_id(&backend, a);
    return backend == activeBackend;
}

[[gnu::unused]] bool checkArray(af_backend activeBackend, const af_array* a) {
    if (a) {
        return checkArray(activeBackend, *a);
    } else {
        return true;
    }
}

[[gnu::unused]] bool checkArrays(af_backend activeBackend) {
    UNUSED(activeBackend);
    // Dummy
    return true;
}

}  // namespace

template<typename T, typename... Args>
bool checkArrays(af_backend activeBackend, T a, Args... arg) {
    return checkArray(activeBackend, a) && checkArrays(activeBackend, arg...);
}

}  // namespace unified
}  // namespace arrayfire

/// Checks if the active backend and the af_arrays are the same.
///
/// Checks if the active backend and the af_array's backend match. If they do
/// not match, an error is returned. This macro accepts pointer to af_arrays
/// and af_arrays. Null pointers to af_arrays are considered acceptable.
///
/// \param[in] Any number of af_arrays or pointer to af_arrays
#define CHECK_ARRAYS(...)                                                     \
    do {                                                                      \
        af_backend backendId = arrayfire::unified::getActiveBackend();        \
        if (!arrayfire::unified::checkArrays(backendId, __VA_ARGS__))         \
            AF_RETURN_ERROR("Input array does not belong to current backend", \
                            AF_ERR_ARR_BKND_MISMATCH);                        \
    } while (0)

#define CALL(FUNCTION, ...)                                                      \
    using af_func                  = std::add_pointer<decltype(FUNCTION)>::type; \
    thread_local af_backend index_ = arrayfire::unified::getActiveBackend();     \
    if (arrayfire::unified::getActiveHandle()) {                                 \
        thread_local af_func func =                                              \
            (af_func)arrayfire::common::getFunctionPointer(                      \
                arrayfire::unified::getActiveHandle(), __func__);                \
        if (!func) {                                                             \
            AF_RETURN_ERROR(                                                     \
                "requested symbol name could not be found in loaded library.",   \
                AF_ERR_LOAD_LIB);                                                \
        }                                                                        \
        if (index_ != arrayfire::unified::getActiveBackend()) {                  \
            index_ = arrayfire::unified::getActiveBackend();                     \
            func   = (af_func)arrayfire::common::getFunctionPointer(             \
                arrayfire::unified::getActiveHandle(), __func__);              \
        }                                                                        \
        return func(__VA_ARGS__);                                                \
    } else {                                                                     \
        AF_RETURN_ERROR("ArrayFire couldn't locate any backends.",               \
                        AF_ERR_LOAD_LIB);                                        \
    }

#define CALL_NO_PARAMS(FUNCTION) CALL(FUNCTION)

#define LOAD_SYMBOL()                      \
    arrayfire::common::getFunctionPointer( \
        arrayfire::unified::getActiveHandle(), __FUNCTION__)
