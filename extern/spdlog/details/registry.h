//
// Copyright(c) 2015 Gabi Melman.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#pragma once

// Loggers registy of unique name->logger pointer
// An attempt to create a logger with an already existing name will be ignored
// If user requests a non existing logger, nullptr will be returned
// This class is thread safe

#include "spdlog/common.h"
#include "spdlog/details/periodic_worker.h"
#include "spdlog/logger.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace spdlog {
namespace details {
class thread_pool;

class registry
{
public:
    registry(const registry &) = delete;
    registry &operator=(const registry &) = delete;

    void register_logger(std::shared_ptr<logger> new_logger)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        auto logger_name = new_logger->name();
        throw_if_exists_(logger_name);
        loggers_[logger_name] = new_logger;
    }

    void register_and_init(std::shared_ptr<logger> new_logger)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        auto logger_name = new_logger->name();
        throw_if_exists_(logger_name);

        // set the global formatter pattern
        new_logger->set_formatter(formatter_->clone());

        if (err_handler_)
        {
            new_logger->set_error_handler(err_handler_);
        }

        new_logger->set_level(level_);
        new_logger->flush_on(flush_level_);

        // add to registry
        loggers_[logger_name] = new_logger;
    }

    std::shared_ptr<logger> get(const std::string &logger_name)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        auto found = loggers_.find(logger_name);
        return found == loggers_.end() ? nullptr : found->second;
    }

    void set_tp(std::shared_ptr<thread_pool> tp)
    {
        std::lock_guard<std::recursive_mutex> lock(tp_mutex_);
        tp_ = std::move(tp);
    }

    std::shared_ptr<thread_pool> get_tp()
    {
        std::lock_guard<std::recursive_mutex> lock(tp_mutex_);
        return tp_;
    }

    // Set global formatter. Each sink in each logger will get a clone of this object
    void set_formatter(std::unique_ptr<formatter> formatter)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        formatter_ = std::move(formatter);
        for (auto &l : loggers_)
        {
            l.second->set_formatter(formatter_->clone());
        }
    }

    void set_level(level::level_enum log_level)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        for (auto &l : loggers_)
        {
            l.second->set_level(log_level);
        }
        level_ = log_level;
    }

    void flush_on(level::level_enum log_level)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        for (auto &l : loggers_)
        {
            l.second->flush_on(log_level);
        }
        flush_level_ = log_level;
    }

    void flush_every(std::chrono::seconds interval)
    {
        std::lock_guard<std::mutex> lock(flusher_mutex_);
        std::function<void()> clbk(std::bind(&registry::flush_all, this));
        periodic_flusher_.reset(new periodic_worker(clbk, interval));
    }

    void set_error_handler(log_err_handler handler)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        for (auto &l : loggers_)
        {
            l.second->set_error_handler(handler);
        }
        err_handler_ = handler;
    }

    void apply_all(std::function<void(std::shared_ptr<logger>)> fun)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        for (auto &l : loggers_)
        {
            fun(l.second);
        }
    }

    void flush_all()
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        for (auto &l : loggers_)
        {
            l.second->flush();
        }
    }

    void drop(const std::string &logger_name)
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        loggers_.erase(logger_name);
    }

    void drop_all()
    {
        std::lock_guard<std::mutex> lock(logger_map_mutex_);
        loggers_.clear();
    }

    // clean all reasources and threads started by the registry
    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(flusher_mutex_);
            periodic_flusher_.reset();
        }

        drop_all();

        {
            std::lock_guard<std::recursive_mutex> lock(tp_mutex_);
            tp_.reset();
        }
    }

    std::recursive_mutex &tp_mutex()
    {
        return tp_mutex_;
    }

    static registry &instance()
    {
        static registry s_instance;
        return s_instance;
    }

private:
    registry()
        : formatter_(new pattern_formatter("%+"))
    {
    }

    ~registry()
    {
        /*std::lock_guard<std::mutex> lock(flusher_mutex_);
        periodic_flusher_.reset();*/
    }

    void throw_if_exists_(const std::string &logger_name)
    {
        if (loggers_.find(logger_name) != loggers_.end())
        {
            throw spdlog_ex("logger with name '" + logger_name + "' already exists");
        }
    }

    std::mutex logger_map_mutex_, flusher_mutex_;
    std::recursive_mutex tp_mutex_;
    std::unordered_map<std::string, std::shared_ptr<logger>> loggers_;
    std::unique_ptr<formatter> formatter_;
    level::level_enum level_ = level::info;
    level::level_enum flush_level_ = level::off;
    log_err_handler err_handler_;
    std::shared_ptr<thread_pool> tp_;
    std::unique_ptr<periodic_worker> periodic_flusher_;
};

} // namespace details
} // namespace spdlog
