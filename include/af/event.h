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

typedef void* af_event;

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_create_event(af_event* eventHandle);

AFAPI af_err af_release_event(const af_event eventHandle);

AFAPI af_err af_mark_event(const af_event eventHandle);

AFAPI af_err af_enqueue_wait_event(const af_event eventHandle);

AFAPI af_err af_block_event(const af_event eventHandle);

#ifdef __cplusplus
}
#endif
