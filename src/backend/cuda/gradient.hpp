/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace cuda {
template<typename T>
void gradient(Array<T> &grad0, Array<T> &grad1, const Array<T> &in);
}  // namespace cuda
}  // namespace arrayfire
