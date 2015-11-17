/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#include <Array.hpp>

namespace cpu
{
    template<typename T>
    void select(Array<T> &out, const Array<char> &cond, const Array<T> &a, const Array<T> &b);

    template<typename T, bool flip>
    void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a, const double &b);
}
