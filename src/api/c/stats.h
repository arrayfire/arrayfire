/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <reduce.hpp>

using namespace detail;

template<typename T>
inline T mean(const Array<T>& in)
{
    return division(reduce_all<af_add_t, T, T>(in), in.elements());
}
