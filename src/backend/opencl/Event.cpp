/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Event.hpp>
#include <af/event.h>
#include <common/err_common.hpp>
#include <events.hpp>
#include <platform.hpp>

namespace opencl {
/// \brief Creates a new event and marks it in the queue
Event make_event(cl::CommandQueue &queue) {
    Event e;
    if (e.create() == CL_SUCCESS) { e.mark(queue()); }
    return e;
}

af_event make_event_on_active_queue() {
  Event *e = new Event();
  if (e->create() == CL_SUCCESS) {
    // Use the currently-active queue
    e->mark(getQueue()());
  } else {
    AF_ERROR("Could not create event", AF_ERR_RUNTIME);
  }
  af_event_t newEvent;
  newEvent.event = e;
  return getEventHandle(newEvent);
}

void mark_event_on_active_queue(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  // Use the currently-active stream
  if (event.event->mark(getQueue()()) != CL_SUCCESS) {
    AF_ERROR("Could not mark event on active queue", AF_ERR_RUNTIME);
  }
}

void release_event(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  delete event.event;
  delete (af_event_t *) eventHandle;
}

void enqueue_wait_on_active_queue(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  // Use the currently-active stream
  if (event.event->enqueueWait(getQueue()()) != CL_SUCCESS) {
    AF_ERROR("Could not enqueue wait on active queue for event",
             AF_ERR_RUNTIME);
  }
}

void block(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  if (event.event->block() != CL_SUCCESS) {
    AF_ERROR("Could not block on active queue for event", AF_ERR_RUNTIME);
  }
}

}  // namespace opencl
