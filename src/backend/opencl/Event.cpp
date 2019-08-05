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
#include <af/event.h>

namespace opencl {
/// \brief Creates a new event and marks it in the queue
Event makeEvent(cl::CommandQueue& queue) {
    Event e;
    if (e.create() == CL_SUCCESS) { e.mark(queue()); }
    return e;
}

af_event createEvent() {
    Event* e     = new Event();
    Event& event = *e;
    // Ensure the default CL command queue is initialized
    getQueue()();
    if (event.create() != CL_SUCCESS) {
        delete e;  // don't leak the event if creation fails
        AF_ERROR("Could not create event", AF_ERR_RUNTIME);
    }
    af_event eventHandle = getHandle(event);
    markEventOnActiveQueue(eventHandle);
    return eventHandle;
}

void releaseEvent(af_event eventHandle) { delete (Event*)eventHandle; }

void markEventOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    if (event.mark(getQueue()()) != CL_SUCCESS) {
        AF_ERROR("Could not mark event on active queue", AF_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    if (event.enqueueWait(getQueue()()) != CL_SUCCESS) {
        AF_ERROR("Could not enqueue wait on active queue for event",
                 AF_ERR_RUNTIME);
    }
}

void block(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    if (event.block() != CL_SUCCESS) {
        AF_ERROR("Could not block on active queue for event", AF_ERR_RUNTIME);
    }
}

}  // namespace opencl
