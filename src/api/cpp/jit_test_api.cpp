/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <jit_test_api.h>
#include "error.hpp"

namespace af {
int getMaxJitLen(void) {
    int retVal = 0;
    AF_THROW(af_get_max_jit_len(&retVal));
    return retVal;
}

void setMaxJitLen(const int jitLen) { AF_THROW(af_set_max_jit_len(jitLen)); }
}  // namespace af
