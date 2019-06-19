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
#include <platform.hpp>

namespace opencl {
class OpenCLEventPolicy {
   public:
    using EventType = cl_event;
    using QueueType = cl_command_queue;
    using ErrorType = cl_int;

    static cl_int createEvent(cl_event *e) noexcept {
        // Events are created when you mark them
        return CL_SUCCESS;
    }

    static cl_int markEvent(cl_event *e, cl_command_queue stream) noexcept {
        return clEnqueueMarkerWithWaitList(stream, 0, nullptr, e);
    }

    static cl_int waitForEvent(cl_event *e, cl_command_queue stream) noexcept {
        return clEnqueueMarkerWithWaitList(stream, 1, e, nullptr);
    }

    static cl_int syncForEvent(cl_event *e) noexcept {
        return clWaitForEvents(1, e);
    }

    static cl_int destroyEvent(cl_event *e) noexcept {
        return clReleaseEvent(*e);
    }
};

using Event = common::EventBase<OpenCLEventPolicy>;

/// \brief Creates a new event and marks it in the queue
Event make_event(cl::CommandQueue &queue);

}  // namespace opencl
