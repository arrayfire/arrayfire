/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <Array.hpp>

namespace cuda
{
    template<typename T>
    Array<T> transform(const Array<T> &in, const Array<float> &tf, const af::dim4 &odims,
                       const af_interp_type method, const bool inverse,
                       const bool perspective);
}
