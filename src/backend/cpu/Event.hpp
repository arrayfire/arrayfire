/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/EventBase.hpp>
#include <queue.hpp>
#include <af/event.h>

#include <type_traits>

namespace arrayfire {
namespace cpu {

class CPUEventPolicy {
   public:
    using EventType = queue_event;
    using QueueType = std::add_lvalue_reference<queue>::type;
    using ErrorType = int;

    static int createAndMarkEvent(queue_event *e) noexcept {
        return e->create();
    }

    static int markEvent(queue_event *e, cpu::queue &stream) noexcept {
        return e->mark(stream);
    }

    static int waitForEvent(queue_event *e, cpu::queue &stream) noexcept {
        return e->wait(stream);
    }

    static int syncForEvent(queue_event *e) noexcept {
        e->sync();
        return 0;
    }

    static int destroyEvent(queue_event *e) noexcept { return 0; }
};

using Event = common::EventBase<CPUEventPolicy>;

/// \brief Creates a new event and marks it in the queue
Event makeEvent(cpu::queue &queue);

af_event createEvent();

void markEventOnActiveQueue(af_event eventHandle);

void enqueueWaitOnActiveQueue(af_event eventHandle);

void block(af_event eventHandle);

af_event createAndMarkEvent();

}  // namespace cpu
}  // namespace arrayfire
