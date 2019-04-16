/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/signal.h>
#include "error.hpp"

namespace af {
array approx1(const array &in, const array &pos, const interpType method,
              const float offGrid) {
    af_array out = 0;
    AF_THROW(af_approx1(&out, in.get(), pos.get(), method, offGrid));
    return array(out);
}

array approx2(const array &in, const array &pos0, const array &pos1,
              const interpType method, const float offGrid) {
    af_array out = 0;
    AF_THROW(
        af_approx2(&out, in.get(), pos0.get(), pos1.get(), method, offGrid));
    return array(out);
}
}  // namespace af
