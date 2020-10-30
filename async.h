
//
// Copyright(c) 2018 Gabi Melman.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#pragma once

//
// Async logging using global thread pool
// All loggers created here share same global thread pool.
// Each log message is pushed to a queue along withe a shared pointer to the
// logger.
// If a logger deleted while having pending messages in the queue, it's actual
// destruction will defer
// until all its messages are processed by the thread pool.
// This is because each message in the queue holds a shared_ptr to the
// originating logger.

#include "spdlog/async_logger.h"
#include "spdlog/details/registry.h"
#include "spdlog/details/thread_pool.h"

#include <memory>
#include <mutex>

namespace spdlog {

namespace details {
static const size_t default_async_q_size = 8192;
}

// async logger factory - creates async loggers backed with thread pool.
// if a global thread pool doesn't already exist, create it with default queue
// size of 8192 items and single thread.
template<async_overflow_policy OverflowPolicy = async_overflow_policy::block>
struct async_factory_impl
{
    template<typename Sink, typename... SinkArgs>
    static std::shared_ptr<async_logger> create(const std::string &logger_name, SinkArgs &&... args)
    {
        auto &registry_inst = details::registry::instance();

        // create global thread pool if not already exists..
        std::lock_guard<std::recursive_mutex>(registry_inst.tp_mutex());
        auto tp = registry_inst.get_tp();
        if (tp == nullptr)
        {
            tp = std::make_shared<details::thread_pool>(details::default_async_q_size, 1);
            registry_inst.set_tp(tp);
        }

        auto sink = std::make_shared<Sink>(std::forward<SinkArgs>(args)...);
        auto new_logger = std::make_shared<async_logger>(logger_name, std::move(sink), std::move(tp), OverflowPolicy);
        registry_inst.register_and_init(new_logger);
        return new_logger;
    }
};

using async_factory = async_factory_impl<async_overflow_policy::block>;
using async_factory_nonblock = async_factory_impl<async_overflow_policy::overrun_oldest>;

template<typename Sink, typename... SinkArgs>
inline std::shared_ptr<spdlog::logger> create_async(const std::string &logger_name, SinkArgs &&... sink_args)
{
    return async_factory::create<Sink>(logger_name, std::forward<SinkArgs>(sink_args)...);
}

template<typename Sink, typename... SinkArgs>
inline std::shared_ptr<spdlog::logger> create_async_nb(const std::string &logger_name, SinkArgs &&... sink_args)
{
    return async_factory_nonblock::create<Sink>(logger_name, std::forward<SinkArgs>(sink_args)...);
}

// set global thread pool.
inline void init_thread_pool(size_t q_size, size_t thread_count)
{
    auto tp = std::make_shared<details::thread_pool>(q_size, thread_count);
    details::registry::instance().set_tp(std::move(tp));
}

// get the global thread pool.
inline std::shared_ptr<spdlog::details::thread_pool> thread_pool()
{
    return details::registry::instance().get_tp();
}
} // namespace spdlog
