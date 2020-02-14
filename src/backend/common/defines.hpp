/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <mutex>
#include <string>

inline std::string clipFilePath(std::string path, std::string str) {
    try {
        std::string::size_type pos = path.rfind(str);
        if (pos == std::string::npos) {
            return path;
        } else {
            return path.substr(pos);
        }
    } catch (...) { return path; }
}

#define UNUSED(expr) \
    do { (void)(expr); } while (0)

#if defined(_WIN32) || defined(_MSC_VER)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#if _MSC_VER < 1900
#define snprintf sprintf_s
#endif
#define STATIC_ static
#define __AF_FILENAME__ (clipFilePath(__FILE__, "src\\").c_str())
#else
//#ifndef __PRETTY_FUNCTION__
//    #define __PRETTY_FUNCTION__ __func__ // __PRETTY_FUNCTION__ Fallback
//#endif
#define STATIC_ inline
#define __AF_FILENAME__ (clipFilePath(__FILE__, "src/").c_str())
#endif

typedef enum {
    AF_BATCH_UNSUPPORTED = -1, /* invalid inputs */
    AF_BATCH_NONE,             /* one signal, one filter   */
    AF_BATCH_LHS,              /* many signal, one filter  */
    AF_BATCH_RHS,              /* one signal, many filter  */
    AF_BATCH_SAME,             /* signal and filter have same batch size */
    AF_BATCH_DIFF,             /* signal and filter have different batch size */
} AF_BATCH_KIND;

enum class kJITHeuristics {
    Pass                = 0, /* no eval necessary */
    TreeHeight          = 1, /* eval due to jit tree height */
    KernelParameterSize = 2, /* eval due to many kernel parameters */
    MemoryPressure      = 3  /* eval due to memory pressure */
};

#ifdef OS_WIN
#include <Windows.h>
using LibHandle = HMODULE;
#define AF_PATH_SEPARATOR "\\"
#elif defined(OS_MAC)
using LibHandle = void*;
#define AF_PATH_SEPARATOR "/"
#elif defined(OS_LNX)
using LibHandle = void*;
#define AF_PATH_SEPARATOR "/"
#else
#error "Unsupported platform"
#endif

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

namespace common {
using mutex_t      = std::mutex;
using lock_guard_t = std::lock_guard<mutex_t>;
}  // namespace common
