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

#include <spdlog/spdlog.h>

namespace common {
std::shared_ptr<spdlog::logger> loggerFactory(std::string name);
std::string bytesToString(size_t bytes);
}  // namespace common

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
