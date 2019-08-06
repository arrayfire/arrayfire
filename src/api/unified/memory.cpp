/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/memory.h>
#include "symbol_manager.hpp"

af_err af_create_buffer_info(af_buffer_info* pair, void* ptr, af_event event) {
    return CALL(pair, ptr, event);
}

af_err af_release_buffer_info(af_buffer_info pair) { return CALL(pair); }

af_err af_buffer_info_get_ptr(void** ptr, af_buffer_info pair) {
    return CALL(ptr, pair);
}

af_err af_buffer_info_get_event(af_event* event, af_buffer_info pair) {
    return CALL(event, pair);
}

af_err af_buffer_info_set_ptr(af_buffer_info pair, void* ptr) {
    return CALL(pair, ptr);
}

af_err af_buffer_info_set_event(af_buffer_info pair, af_event event) {
    return CALL(pair, event);
}
