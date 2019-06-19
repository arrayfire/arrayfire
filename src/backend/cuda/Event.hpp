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

namespace cuda {

class CUDARuntimeEventPolicy {
   public:
    using EventType = CUevent;
    using QueueType = CUstream;
    using ErrorType = CUresult;

    static ErrorType createEvent(CUevent *e) noexcept {
      auto err = cuEventCreate(e, CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC);
        // printf("create %p: error: %s\n", *e, cudaGetErrorName(err));
        return err;
    }

    static ErrorType markEvent(CUevent *e,
                               QueueType &stream) noexcept {
        auto err = cuEventRecord(*e, stream);
        // printf("mark   %p: error: %s\n", *e, cudaGetErrorName(err));
        return err;
    }

    static ErrorType waitForEvent(CUevent *e,
                                  QueueType &stream) noexcept {
      auto err = cuStreamWaitEvent(stream, *e, 0);
        // printf("wait   %p: error: %s\n", *e, cudaGetErrorName(err));
        return err;
    }

    static ErrorType syncForEvent(CUevent *e) noexcept {
        return cuEventSynchronize(*e);
    }

    static ErrorType destroyEvent(CUevent *e) noexcept {
        auto err = cuEventDestroy(*e);
        // printf("destroy %p: error: %s\n", *e, cudaGetErrorName(err));
        return err;
    }
};

using Event = common::EventBase<CUDARuntimeEventPolicy>;

/// \brief Creates a new event and marks it in the stream
Event make_event(cudaStream_t stream);

}  // namespace cuda
