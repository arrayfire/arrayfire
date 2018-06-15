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

#ifdef AF_WITH_LOGGING
#include <spdlog/spdlog.h>
#else

/// This is a stub class to match the spdlog API in case it is not installed on
/// the users system. Only the functions we used are implemented here. Other
/// functions will need to be implemented later.
namespace spdlog {
    class logger { public: logger() {} };
    std::shared_ptr<spdlog::logger> get(std::string &name);
    std::shared_ptr<spdlog::logger> stdout_logger_mt(std::string&);
    namespace level {
        enum enum_level { trace };
    }
}
#endif

namespace common {
    std::shared_ptr<spdlog::logger> loggerFactory(std::string name);
    std::string bytesToString(size_t bytes);
}

#ifdef AF_WITH_LOGGING
#define AF_STR_H(x) #x
#define AF_STR_HELPER(x) AF_STR_H(x)
#ifdef _MSC_VER
#define AF_TRACE(...) getLogger()->trace("[ " __FILE__ "(" AF_STR_HELPER(__LINE__) ") ] " __VA_ARGS__)
#else
#define AF_TRACE(...) getLogger()->trace("[ " __FILE__ ":" AF_STR_HELPER(__LINE__) " ] " __VA_ARGS__)
#endif
#else
#define AF_TRACE(logger, ...) (void)0
#endif
