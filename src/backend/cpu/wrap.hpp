/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
void wrap(Array<T> &out, const Array<T> &in, const dim_t wx, const dim_t wy,
          const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
          const bool is_column);

template<typename T>
Array<T> wrap_dilated(const Array<T> &in, const dim_t ox, const dim_t oy,
                      const dim_t wx, const dim_t wy, const dim_t sx,
                      const dim_t sy, const dim_t px, const dim_t py,
                      const dim_t dx, const dim_t dy, const bool is_column);
}  // namespace cpu
}  // namespace arrayfire
