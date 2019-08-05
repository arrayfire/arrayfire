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
#include <cuda_runtime_api.h>
#include <events.hpp>
#include <platform.hpp>
#include <af/event.h>

namespace cuda {
/// \brief Creates a new event and marks it in the queue
Event make_event(cudaStream_t queue) {
    Event e;
    if (e.create() == CUDA_SUCCESS) { e.mark(queue); }
    return e;
}

af_event createEvent() {
    Event* e     = new Event();
    Event& event = *e;
    // Default CUDA stream needs to be initialized to use the CUDA driver Ctx
    getActiveStream();
    if (event.create() != CUDA_SUCCESS) {
        AF_ERROR("Could not create event", AF_ERR_RUNTIME);
    }
    af_event eventHandle = getEventHandle(event);
    markEventOnActiveQueue(eventHandle);
    return eventHandle;
}

void releaseEvent(af_event eventHandle) { delete (Event*)eventHandle; }

void markEventOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    cudaStream_t stream = getActiveStream();
    if (event.mark(stream) != CUDA_SUCCESS) {
        AF_ERROR("Could not mark event on active stream", AF_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    cudaStream_t stream = getActiveStream();
    if (event.enqueueWait(stream) != CUDA_SUCCESS) {
        AF_ERROR("Could not enqueue wait on active stream for event",
                 AF_ERR_RUNTIME);
    }
}

void block(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    if (event.block() != CUDA_SUCCESS) {
        AF_ERROR("Could not block on active stream for event", AF_ERR_RUNTIME);
    }
}

}  // namespace cuda
