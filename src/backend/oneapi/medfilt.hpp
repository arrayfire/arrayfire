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
Array<T> medfilt1(const Array<T> &in, const int w_wid,
                  const af::borderType edge_pad);

template<typename T>
Array<T> medfilt2(const Array<T> &in, const int w_len, const int w_wid,
                  const af::borderType edge_pad);

}  // namespace oneapi
}  // namespace arrayfire
