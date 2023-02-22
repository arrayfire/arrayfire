/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/EventBase.hpp>

#include <af/event.h>

#include <sycl/event.hpp>
#include <sycl/queue.hpp>

namespace arrayfire {
namespace oneapi {
class OneAPIEventPolicy {
   public:
    using EventType = sycl::event;
    using QueueType = sycl::queue;
    // using ErrorType = sycl::exception; //does this make sense
    using ErrorType = int;

    static ErrorType createAndMarkEvent(EventType *e) noexcept {
        // Events are created when you mark them
        return 0;
    }

    static ErrorType markEvent(EventType *e, QueueType stream) noexcept {
        // return clEnqueueMarkerWithWaitList(stream, 0, nullptr, e);
        return 0;
    }

    static ErrorType waitForEvent(EventType *e, QueueType stream) noexcept {
        // return clEnqueueMarkerWithWaitList(stream, 1, e, nullptr);
        return 0;
    }

    static ErrorType syncForEvent(EventType *e) noexcept {
        // return clWaitForEvents(1, e);
        return 0;
    }

    static ErrorType destroyEvent(EventType *e) noexcept {
        // return clReleaseEvent(*e);
        return 0;
    }
};

using Event = common::EventBase<OneAPIEventPolicy>;

/// \brief Creates a new event and marks it in the queue
Event makeEvent(sycl::queue &queue);

af_event createEvent();

void markEventOnActiveQueue(af_event eventHandle);

void enqueueWaitOnActiveQueue(af_event eventHandle);

void block(af_event eventHandle);

af_event createAndMarkEvent();

}  // namespace oneapi
}  // namespace arrayfire
