/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <vector>

namespace arrayfire {
namespace cpu {
template<typename T>
Array<T> join(const int dim, const Array<T> &first, const Array<T> &second);

template<typename T>
void join(Array<T> &output, const int dim, const std::vector<Array<T>> &inputs);
}  // namespace cpu
}  // namespace arrayfire
