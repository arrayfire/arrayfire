/*******************************************************
 * Copyright (c) 2018, ArrayFire
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
void topk(Array<T>& keys, Array<unsigned>& vals, const Array<T>& in,
          const int k, const int dim, const af::topkFunction order);
}  // namespace cuda
}  // namespace arrayfire
