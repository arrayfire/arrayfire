/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace threads {

class async_queue {
    std::thread queue_thread;
    std::queue<std::function<void()>> work_queue;
    std::atomic<bool> done;
    std::mutex queue_mutex;
    std::mutex work_mutex;
    std::condition_variable cond;

    void queue_runner() {
        //std::cout << "Starting: " << std::this_thread::get_id() << std::endl;
        while (!done) {
          std::unique_lock<std::mutex> work_lock(work_mutex, std::defer_lock);
          std::function<void()> func;
            {
              std::unique_lock<std::mutex> lock(queue_mutex);
                cond.wait(lock, [this]() {
                        return work_queue.empty() == false || done;
                    });

                if(done) break;

                work_lock.lock();
                swap(func, work_queue.front());
                work_queue.pop();
            }
            if (!func) {printf("bad function: %zu\n", work_queue.size());}
            else func();
        }
    }

public:
    /// \brief Enqueues a new function onto the work queue
    ///
    /// \param func A function which will be enqueued on the work queue
    /// \param args The argument of the funciton \p func
    template <typename F, typename... Args>
    void enqueue(const F func, Args... args) {
        if(std::this_thread::get_id() == queue_thread.get_id()) {
            func(args...);
        } else {
            auto no_arg_func = std::bind(func, std::forward<Args>(args)...);
            {
              std::lock_guard<std::mutex> lock(queue_mutex);
                work_queue.push(no_arg_func);
            }

            cond.notify_one();
        }
        return;
    }

    /// \brief Check if the current thread of execution is same as the queue thread
    ///
    /// \return A boolean indicating if current thread is same as the queue thread
    bool is_worker() const {
        return std::this_thread::get_id() == queue_thread.get_id();
    }

    /// \brief Blocks until all work has completed
    ///
    /// This function will block the calling thread until all of the queued
    /// functions have completed
    void sync() {
        //std::cout << "Syncing" << std::endl;
        std::promise<void> p;
        std::future<void> fut = p.get_future();
        auto f = [] (std::promise<void>& pr) {
            pr.set_value();
        };
        this->enqueue(f, ref(p));
        fut.wait();
        //std::cout << "Done Syncing" << std::endl;
    }

    /// \brief Creates a new work queue
    async_queue()
        : work_queue()
        , done(false) {
      std::thread tmp(&async_queue::queue_runner, this);
      std::swap(queue_thread, tmp);
    }

    ~async_queue() {
        done.store(true);
        cond.notify_one();
        queue_thread.join();
    }
};

}  // namespace threads
