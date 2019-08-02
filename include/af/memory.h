/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <af/event.h>

typedef void* af_memory_event_pair;

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_create_memory_event_pair(
    af_memory_event_pair* pair,
    void* ptr,
    af_event event);

AFAPI af_err af_release_memory_event_pair(af_memory_event_pair pair);

AFAPI af_err af_memory_event_pair_set_ptr(af_memory_event_pair pair, void* ptr);

AFAPI af_err
af_memory_event_pair_set_event(af_memory_event_pair pairHandle, af_event event);

AFAPI af_err
af_memory_event_pair_get_ptr(void** ptr, af_memory_event_pair pair);

AFAPI af_err
af_memory_event_pair_get_event(af_event* event, af_memory_event_pair pair);

#ifdef __cplusplus
}
#endif
