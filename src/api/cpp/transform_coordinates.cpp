/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "error.hpp"

namespace af {

array transformCoordinates(const array& tf, const float d0, const float d1) {
    af_array out = 0;
    AF_THROW(af_transform_coordinates(&out, tf.get(), d0, d1));
    return array(out);
}

}  // namespace af
