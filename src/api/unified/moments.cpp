/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "symbol_manager.hpp"

af_err af_moments(af_array* out, const af_array in,
                  const af_moment_type moment) {
    CHECK_ARRAYS(in);
    CALL(af_moments, out, in, moment);
}

af_err af_moments_all(double* out, const af_array in,
                      const af_moment_type moment) {
    CHECK_ARRAYS(in);
    CALL(af_moments_all, out, in, moment);
}
