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
#include <cuda_runtime_api.h>
#include <events.hpp>
#include <platform.hpp>

namespace cuda {
/// \brief Creates a new event and marks it in the queue
Event make_event(cudaStream_t queue) {
  Event e;
  if (e.create() == CUDA_SUCCESS) {
    e.mark(queue);
  }
  return e;
}

af_event make_event_on_active_queue() {
  Event *e = new Event();
  if (e->create() == CUDA_SUCCESS) {
    // Use the currently-active stream
    cudaStream_t stream = getActiveStream();
    e->mark(stream);
  } else {
    AF_ERROR("Could not create event", AF_ERR_RUNTIME);
  }
  af_event_t newEvent;
  newEvent.event = e;
  return getEventHandle(newEvent);
}

void release_event(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  delete event.event;
  delete (af_event_t *) eventHandle;
}

void mark_event_on_active_queue(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  // Use the currently-active stream
  cudaStream_t stream = getActiveStream();
  if (event.event->mark(stream) != CUDA_SUCCESS) {
    AF_ERROR("Could not mark event on active stream", AF_ERR_RUNTIME);
  }
}

void enqueue_wait_on_active_queue(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  // Use the currently-active stream
  cudaStream_t stream = getActiveStream();
  if (event.event->enqueueWait(stream) != CUDA_SUCCESS) {
    AF_ERROR("Could not enqueue wait on active stream for event",
             AF_ERR_RUNTIME);
  }
}

void block(af_event eventHandle) {
  af_event_t event = getEvent(eventHandle);
  if (event.event->block() != CUDA_SUCCESS) {
    AF_ERROR("Could not block on active stream for event", AF_ERR_RUNTIME);
  }
}

} // namespace cuda
