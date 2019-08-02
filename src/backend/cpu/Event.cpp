/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Event.hpp>

#include <common/err_common.hpp>
#include <events.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/event.h>

namespace cpu {
/// \brief Creates a new event and marks it in the queue
Event make_event(cpu::queue &queue) {
    Event e;
    if (0 == e.create()) { e.mark(queue); }
    return e;
}

af_event makeEventOnActiveQueue() {
    Event *e = new Event();
    if (e->create() == 0) {
        // Use the currently-active queue
        e->mark(getQueue());
    } else {
        AF_ERROR("Could not create event", AF_ERR_RUNTIME);
    }
    af_event_t newEvent;
    newEvent.event = e;
    return getEventHandle(newEvent);
}

void releaseEvent(af_event eventHandle) {
    af_event_t event = getEvent(eventHandle);
    delete event.event;
    delete (af_event_t *)eventHandle;
}

void markEventOnActiveQueue(af_event eventHandle) {
    af_event_t event = getEvent(eventHandle);
    // Use the currently-active queue
    if (event.event->mark(getQueue()) != 0) {
        AF_ERROR("Could not mark event on active queue", AF_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(af_event eventHandle) {
    af_event_t event = getEvent(eventHandle);
    // Use the currently-active queue
    if (event.event->enqueueWait(getQueue()) != 0) {
        AF_ERROR("Could not enqueue wait on active queue for event",
                 AF_ERR_RUNTIME);
    }
}

void block(af_event eventHandle) {
    af_event_t event = getEvent(eventHandle);
    if (event.event->block() != 0) {
        AF_ERROR("Could not block on active queue for event", AF_ERR_RUNTIME);
    }
}

}  // namespace cpu
