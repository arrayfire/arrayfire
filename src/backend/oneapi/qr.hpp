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
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &orig);

template<typename T>
Array<T> qr_inplace(Array<T> &in);
}  // namespace oneapi
}  // namespace arrayfire
