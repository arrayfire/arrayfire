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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <af/event.h>

namespace arrayfire {
namespace cuda {

class CUDARuntimeEventPolicy {
   public:
    using EventType = CUevent;
    using QueueType = CUstream;
    using ErrorType = CUresult;

    static ErrorType createAndMarkEvent(CUevent *e) noexcept {
        // Creating events with the CU_EVENT_BLOCKING_SYNC flag
        // severly impacts the speed if/when creating many arrays
        auto err = cuEventCreate(e, CU_EVENT_DISABLE_TIMING);
        return err;
    }

    static ErrorType markEvent(CUevent *e, QueueType &stream) noexcept {
        auto err = cuEventRecord(*e, stream);
        return err;
    }

    static ErrorType waitForEvent(CUevent *e, QueueType &stream) noexcept {
        auto err = cuStreamWaitEvent(stream, *e, 0);
        return err;
    }

    static ErrorType syncForEvent(CUevent *e) noexcept {
        return cuEventSynchronize(*e);
    }

    static ErrorType destroyEvent(CUevent *e) noexcept {
        auto err = cuEventDestroy(*e);
        return err;
    }
};

using Event = common::EventBase<CUDARuntimeEventPolicy>;

/// \brief Creates a new event and marks it in the stream
Event makeEvent(cudaStream_t queue);

af_event createEvent();

void markEventOnActiveQueue(af_event eventHandle);

void enqueueWaitOnActiveQueue(af_event eventHandle);

void block(af_event eventHandle);

af_event createAndMarkEvent();

}  // namespace cuda
}  // namespace arrayfire
