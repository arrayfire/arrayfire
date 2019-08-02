/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/event.h>
#include <af/memory.h>

typedef struct {
  void *ptr;
  af_event event;
} af_memory_event_pair_t;

af_memory_event_pair_t getMemoryEventPair(const af_memory_event_pair pair);

af_memory_event_pair
getMemoryEventPairHandle(const af_memory_event_pair_t pairHandle);
