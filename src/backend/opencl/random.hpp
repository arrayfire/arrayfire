/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <Array.hpp>

namespace opencl
{
    template<typename T>
    Array<T> randu(const af::dim4 &dims);

    template<typename T>
    Array<T> randn(const af::dim4 &dims);

    void setSeed(const uintl seed);
    uintl getSeed();
}
