/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/defines.h>
#include <af/image.h>
#include "error.hpp"

namespace af {

array moments(const array& in, const af_moment_type moment) {
    af_array out = 0;
    AF_THROW(af_moments(&out, in.get(), moment));
    return array(out);
}

void moments(double* out, const array& in, const af_moment_type moment) {
    AF_THROW(af_moments_all(out, in.get(), moment));
}

}  // namespace af
