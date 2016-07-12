/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <af/defines.h>
#include <backend.hpp>

namespace opencl
{
    template<typename T>
    Array<T> uniformDistribution(const af::dim4 &dims, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
}
