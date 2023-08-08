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

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
class OneAPIEventPolicy {
   public:
    using EventType = sycl::event *;
    using QueueType = sycl::queue;
    using ErrorType = int;

    static ErrorType createAndMarkEvent(EventType *e) noexcept {
        *e = new sycl::event;
        return 0;
    }

    static ErrorType markEvent(EventType *e, QueueType stream) noexcept {
        **e = stream.ext_oneapi_submit_barrier();
        return 0;
    }

    static ErrorType waitForEvent(EventType *e, QueueType stream) noexcept {
        stream.ext_oneapi_submit_barrier({**e});
        return 0;
    }

    static ErrorType syncForEvent(EventType *e) noexcept {
        (*e)->wait();
        return 0;
    }

    static ErrorType destroyEvent(EventType *e) noexcept {
        delete *e;
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
