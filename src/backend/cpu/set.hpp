/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
Array<T> setUnique(const Array<T> &in, const bool is_sorted);

template<typename T>
Array<T> setUnion(const Array<T> &first, const Array<T> &second,
                  const bool is_unique);

template<typename T>
Array<T> setIntersect(const Array<T> &first, const Array<T> &second,
                      const bool is_unique);
}  // namespace cpu
}  // namespace arrayfire
