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

array regions(const array& in, const af::connectivity connectivity,
              const af::dtype type) {
    af_array temp = 0;
    AF_THROW(af_regions(&temp, in.get(), connectivity, type));
    return array(temp);
}

}  // namespace af
