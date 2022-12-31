/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace oneapi {
template<typename T>
Array<T> diagCreate(const Array<T> &in, const int num);

template<typename T>
Array<T> diagExtract(const Array<T> &in, const int num);
}  // namespace oneapi
}  // namespace arrayfire
