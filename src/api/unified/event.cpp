/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/event.h>
#include "symbol_manager.hpp"

af_err af_create_event(af_event* eventHandle) {
    CALL(af_create_event, eventHandle);
}

af_err af_delete_event(af_event eventHandle) {
    CALL(af_delete_event, eventHandle);
}

af_err af_mark_event(const af_event eventHandle) {
    CALL(af_mark_event, eventHandle);
}

af_err af_enqueue_wait_event(const af_event eventHandle) {
    CALL(af_enqueue_wait_event, eventHandle);
}

af_err af_block_event(const af_event eventHandle) {
    CALL(af_block_event, eventHandle);
}
