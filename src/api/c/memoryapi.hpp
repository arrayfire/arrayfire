/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

struct MemoryEventPair {
    void *ptr;
    af_event event;
};

MemoryEventPair &getMemoryEventPair(const af_memory_event_pair pair);

af_memory_event_pair getMemoryEventPairHandle(MemoryEventPair &pairHandle);
