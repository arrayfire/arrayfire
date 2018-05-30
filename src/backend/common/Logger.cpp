
#include <common/Logger.hpp>
#include <common/util.hpp>

#include <array>
#include <cstdlib>
#include <string>
#include <memory>

using std::array;
using std::make_shared;
using std::string;
using std::shared_ptr;
using std::to_string;

using spdlog::level::trace;
using spdlog::logger;
using spdlog::stdout_logger_mt;

namespace common {

#ifdef AF_WITH_LOGGING
shared_ptr<logger>
loggerFactory(string name) {
    auto logger = stdout_logger_mt(name);
    logger->set_pattern("[%n][%t] %v");

    // Log mode
    string env_var = getEnvVar("AF_TRACE");
    if(env_var.find_first_of("all") != string::npos ||
       env_var.find_first_of(name) != string::npos)
        logger->set_level(trace);
    return logger;
}

string bytesToString(size_t bytes) {
  static array<const char *, 5> units{"B", "KB", "MB", "GB", "TB"};
  int count = 0;
  double fbytes = static_cast<double>(bytes);
  for(count = 0; count < units.size() && fbytes > 1000.0; count++) {
    fbytes *= (1.0 / 1024.0);
  }
  return fmt::format("{:.3g} {}", fbytes, units[count]);
}
#else
  shared_ptr<logger>
  loggerFactory(string name) {
    return make_shared<logger>();
  }

  string bytesToString(size_t bytes) {
    return "";
  }
#endif
}
