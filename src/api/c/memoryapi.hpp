/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <events.hpp>
#include <af/event.h>
#include <af/memory.h>

struct BufferInfo {
    void *ptr;
    af_event event;
};

BufferInfo &getBufferInfo(const af_buffer_info pair);

af_buffer_info getHandle(BufferInfo &pairHandle);
