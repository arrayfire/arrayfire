/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Event.hpp>

#include <err_oneapi.hpp>
#include <events.hpp>
#include <platform.hpp>
#include <af/event.h>
#include <memory>

#include <memory>

using std::make_unique;
using std::unique_ptr;

namespace arrayfire {
namespace oneapi {
/// \brief Creates a new event and marks it in the queue
Event makeEvent(sycl::queue& queue) {
    Event e;
    if (e.create() == 0) { e.mark(queue); }
    return e;
}

af_event createEvent() {
    auto e = make_unique<Event>();
    // Ensure the default CL command queue is initialized
    getQueue();
    if (e->create() != 0) {
        AF_ERROR("Could not create event", AF_ERR_RUNTIME);
    }
    Event& ref = *e.release();
    return getHandle(ref);
}

void markEventOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    if (event.mark(getQueue()) != 0) {
        AF_ERROR("Could not mark event on active queue", AF_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    if (event.enqueueWait(getQueue()) != 0) {
        AF_ERROR("Could not enqueue wait on active queue for event",
                 AF_ERR_RUNTIME);
    }
}

void block(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    if (event.block() != 0) {
        AF_ERROR("Could not block on active queue for event", AF_ERR_RUNTIME);
    }
}

af_event createAndMarkEvent() {
    af_event handle = createEvent();
    markEventOnActiveQueue(handle);
    return handle;
}

}  // namespace oneapi
}  // namespace arrayfire
