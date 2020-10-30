//
// Copyright (c) 2015 David Schury, Gabi Melman
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#pragma once

#include "base_sink.h"
#include "spdlog/details/log_msg.h"
#include "spdlog/details/null_mutex.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

// Distribution sink (mux). Stores a vector of sinks which get called when log
// is called

namespace spdlog {
namespace sinks {

template<typename Mutex>
class dist_sink : public base_sink<Mutex>
{
public:
    dist_sink() = default;
    dist_sink(const dist_sink &) = delete;
    dist_sink &operator=(const dist_sink &) = delete;

    void add_sink(std::shared_ptr<sink> sink)
    {
        std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
        sinks_.push_back(sink);
    }

    void remove_sink(std::shared_ptr<sink> sink)
    {
        std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
        sinks_.erase(std::remove(sinks_.begin(), sinks_.end(), sink), sinks_.end());
    }

protected:
    void sink_it_(const details::log_msg &msg) override
    {

        for (auto &sink : sinks_)
        {
            if (sink->should_log(msg.level))
            {
                sink->log(msg);
            }
        }
    }

    void flush_() override
    {
        for (auto &sink : sinks_)
            sink->flush();
    }
    std::vector<std::shared_ptr<sink>> sinks_;
};

using dist_sink_mt = dist_sink<std::mutex>;
using dist_sink_st = dist_sink<details::null_mutex>;

} // namespace sinks
} // namespace spdlog
