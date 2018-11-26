/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifdef _WIN32
#include <windows.h> // spdlog needs this
#endif

#include <common/Logger.hpp>
#include <common/util.hpp>

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <spdlog/sinks/stdout_sinks.h>

using std::array;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::to_string;

using spdlog::get;
using spdlog::level::trace;
using spdlog::logger;
using spdlog::stdout_logger_mt;

namespace common {
shared_ptr<logger>
loggerFactory(string name) {
    shared_ptr<logger> logger;
    if(!(logger = get(name))) {
        logger = stdout_logger_mt(name);
        logger->set_pattern("[%n][%t] %v");

        // Log mode
        string env_var = getEnvVar("AF_TRACE");
        if(env_var.find("all") != string::npos ||
           env_var.find(name) != string::npos) {
          logger->set_level(trace);
        }
    }
    return logger;
}

string bytesToString(size_t bytes) {
    static array<const char *, 5> units{{"B", "KB", "MB", "GB", "TB"}};
    size_t count = 0;
    double fbytes = static_cast<double>(bytes);
    size_t num_units = units.size();
    for(count = 0; count < num_units && fbytes > 1000.0f; count++) {
        fbytes *= (1.0f / 1024.0f);
    }
    return fmt::format("{:.3g} {}", fbytes, units[count]);
}
}
