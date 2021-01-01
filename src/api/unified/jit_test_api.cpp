/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <jit_test_api.h>

#include "symbol_manager.hpp"

af_err af_get_max_jit_len(int *jitLen) { CALL(af_get_max_jit_len, jitLen); }

af_err af_set_max_jit_len(const int jitLen) {
    CALL(af_set_max_jit_len, jitLen);
}
