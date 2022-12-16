/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifdef _WIN32
#include <windows.h>  // spdlog needs this
#endif

#include <common/Logger.hpp>
#include <common/util.hpp>

#include <spdlog/sinks/stdout_sinks.h>
#include <array>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

using std::array;
using std::shared_ptr;
using std::string;

using spdlog::get;
using spdlog::logger;
using spdlog::stdout_logger_mt;

namespace arrayfire {
namespace common {

shared_ptr<logger> loggerFactory(const string& name) {
    shared_ptr<logger> logger;
    if (!(logger = get(name))) {
        logger = stdout_logger_mt(name);
        logger->set_pattern("[%n][%E][%t] %v");

        // Log mode
        string env_var = getEnvVar("AF_TRACE");
        if (env_var.find("all") != string::npos ||
            env_var.find(name) != string::npos) {
            logger->set_level(spdlog::level::trace);
        } else {
            logger->set_level(spdlog::level::off);
        }
    }
    return logger;
}

string bytesToString(size_t bytes) {
    constexpr array<const char*, 7> units{
        {"B", "KB", "MB", "GB", "TB", "PB", "EB"}};
    size_t count     = 0;
    auto fbytes      = static_cast<double>(bytes);
    size_t num_units = units.size();
    for (count = 0; count < num_units && fbytes > 1000.0f; count++) {
        fbytes *= (1.0f / 1024.0f);
    }
    if (count == units.size()) { count--; }
    return fmt::format("{:.3g} {}", fbytes, units[count]);
}
}  // namespace common
}  // namespace arrayfire
