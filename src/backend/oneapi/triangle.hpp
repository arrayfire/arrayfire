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
void triangle(Array<T> &out, const Array<T> &in, const bool is_upper,
              const bool is_unit_diag);

template<typename T>
Array<T> triangle(const Array<T> &in, const bool is_upper,
                  const bool is_unit_diag);
}  // namespace oneapi
}  // namespace arrayfire
