/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <memory>
#include <string>
#include <type_traits>

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC */
// Fix the warning code here, if any
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
/* GNU GCC/G++ */
#elif defined(_MSC_VER)
/* Microsoft Visual Studio */
#else
/* Other */
#endif

#include <spdlog/spdlog.h>

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic pop
#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC */
// Fix the warning code here, if any
#elif defined(__GNUC__) || defined(__GNUG__)
/* GNU GCC/G++ */
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
/* Microsoft Visual Studio */
#pragma warning(pop)
#else
/* Other */
#endif

namespace arrayfire {
namespace common {
std::shared_ptr<spdlog::logger> loggerFactory(const std::string& name);
std::string bytesToString(size_t bytes);
}  // namespace common
}  // namespace arrayfire

#ifdef AF_WITH_LOGGING
#define AF_STR_H(x) #x
#define AF_STR_HELPER(x) AF_STR_H(x)
#ifdef _MSC_VER
#define AF_TRACE(...)                \
    getLogger()->trace("[ " __FILE__ \
                       "(" AF_STR_HELPER(__LINE__) ") ] " __VA_ARGS__)
#else
#define AF_TRACE(...)                \
    getLogger()->trace("[ " __FILE__ \
                       ":" AF_STR_HELPER(__LINE__) " ] " __VA_ARGS__)
#endif
#else
#define AF_TRACE(logger, ...) (void)0
#endif
