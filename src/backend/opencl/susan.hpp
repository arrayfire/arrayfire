/*******************************************************
 * Copyright (c) 2015, Arrayfire
 * all rights reserved.
 *
 * This file is distributed under 3-clause bsd license.
 * the complete license agreement can be obtained at:
 * http://Arrayfire.com/licenses/bsd-3-clause
 ********************************************************/

#include <Array.hpp>
#include <af/features.h>

using af::features;

namespace arrayfire {
namespace opencl {

template<typename T>
unsigned susan(Array<float> &x_out, Array<float> &y_out,
               Array<float> &score_out, const Array<T> &in,
               const unsigned radius, const float diff_thr,
               const float geom_thr, const float feature_ratio,
               const unsigned edge);

}  // namespace opencl
}  // namespace arrayfire
