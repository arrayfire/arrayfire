//
// Copyright(c) 2015 Gabi Melman.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#pragma once

#include "spdlog/details/null_mutex.h"
#include "spdlog/sinks/base_sink.h"

#include <mutex>
#include <ostream>

namespace spdlog {
namespace sinks {
template<typename Mutex>
class ostream_sink SPDLOG_FINAL : public base_sink<Mutex>
{
public:
    explicit ostream_sink(std::ostream &os, bool force_flush = false)
        : ostream_(os)
        , force_flush_(force_flush)
    {
    }
    ostream_sink(const ostream_sink &) = delete;
    ostream_sink &operator=(const ostream_sink &) = delete;

protected:
    void sink_it_(const details::log_msg &msg) override
    {
        fmt::memory_buffer formatted;
        sink::formatter_->format(msg, formatted);
        ostream_.write(formatted.data(), formatted.size());
        if (force_flush_)
            ostream_.flush();
    }

    void flush_() override
    {
        ostream_.flush();
    }

    std::ostream &ostream_;
    bool force_flush_;
};

using ostream_sink_mt = ostream_sink<std::mutex>;
using ostream_sink_st = ostream_sink<details::null_mutex>;

} // namespace sinks
} // namespace spdlog
