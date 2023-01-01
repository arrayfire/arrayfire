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
Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x);
}  // namespace oneapi
}  // namespace arrayfire
