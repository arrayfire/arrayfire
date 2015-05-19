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

namespace opencl
{
    template<typename T>
    Array<T> resize(const Array<T> &in, const dim_t odim0, const dim_t odim1,
                    const af_interp_type method);
}
