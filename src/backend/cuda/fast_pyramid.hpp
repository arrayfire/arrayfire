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

#include <vector>

namespace arrayfire {
namespace cuda {
template<typename T>
void fast_pyramid(std::vector<unsigned> &feat_pyr,
                  std::vector<Array<float>> &d_x_pyr,
                  std::vector<Array<float>> &d_y_pyr,
                  std::vector<unsigned> &lvl_best, std::vector<float> &lvl_scl,
                  std::vector<Array<T>> &img_pyr, const Array<T> &in,
                  const float fast_thr, const unsigned max_feat,
                  const float scl_fctr, const unsigned levels,
                  const unsigned patch_size);
}  // namespace cuda
}  // namespace arrayfire
