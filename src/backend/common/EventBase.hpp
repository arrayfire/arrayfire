/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#include <utility>

namespace arrayfire {
namespace common {

template<typename NativeEventPolicy>
class EventBase {
    using QueueType = typename NativeEventPolicy::QueueType;
    using EventType = typename NativeEventPolicy::EventType;
    using ErrorType = typename NativeEventPolicy::ErrorType;
    EventType e_;

   public:
    /// Default constructor of the Event object. Does not create the event.
    constexpr EventBase() noexcept : e_() {}

    /// Deleted copy constructor
    ///
    /// The event object can only be moved.
    EventBase(EventBase &other) = delete;

    /// \brief Move constructor of the Event object. Resets the moved object to
    ///        an invalid event.
    EventBase(EventBase &&other) noexcept
        : e_(std::forward<EventType>(other.e_)) {
        other.e_ = 0;
    }

    /// \brief Event destructor. Calls the destroy event call on the native API
    ~EventBase() noexcept {
        if (e_) NativeEventPolicy::destroyEvent(&e_);
    }

    /// \brief Creates the event object by calling the native create API
    ErrorType create() noexcept {
        return NativeEventPolicy::createAndMarkEvent(&e_);
    }

    /// \brief Adds the event on the queue. Once this point on the program
    ///        is executed, the event is marked complete.
    ///
    /// \returns the error code for the mark call
    ErrorType mark(QueueType queue) noexcept {
        return NativeEventPolicy::markEvent(&e_, queue);
    }

    /// \brief This is an asynchronous function which will block the
    ///        queue/stream from progressing before continuing forward. It will
    ///        not block the calling thread.
    ///
    /// \param queue The queue that will wait for the previous tasks to complete
    ///
    /// \returns the error code for the wait call
    ErrorType enqueueWait(QueueType queue) noexcept {
        return NativeEventPolicy::waitForEvent(&e_, queue);
    }

    /// \brief This function will block the calling thread until the event has
    ///        completed
    ErrorType block() noexcept { return NativeEventPolicy::syncForEvent(&e_); }

    /// \brief Returns true if the event is a valid event.
    constexpr operator bool() const { return e_; }

    EventBase &operator=(EventBase &other) = delete;

    EventBase &operator=(EventBase &&other) noexcept {
        e_       = std::move(other.e_);
        other.e_ = 0;
        return *this;
    }
};

}  // namespace common
}  // namespace arrayfire
