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

namespace arrayfire {
namespace cpu {
template<typename T>
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b);

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const double &b);

template<typename T>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const Array<T> &b, const af::dim4 &odims) {
    Array<T> out = createEmptyArray<T>(odims);
    select(out, cond, a, b);
    return out;
}

template<typename T, bool flip>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const double &b, const af::dim4 &odims) {
    Array<T> out = createEmptyArray<T>(odims);
    select_scalar<T, flip>(out, cond, a, b);
    return out;
}
}  // namespace cpu
}  // namespace arrayfire
