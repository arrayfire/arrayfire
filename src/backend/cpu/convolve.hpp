/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/defines.hpp>

namespace cpu {

template<typename T, typename accT, int baseDim, bool expand>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  AF_BATCH_KIND kind);

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &c_filter,
                   Array<accT> const &r_filter);

template<typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation);

template<typename T>
Array<T> conv2DataGradient(const Array<T> &incoming_gradient,
                           const Array<T> &original_signal,
                           const Array<T> &original_filter,
                           const Array<T> &convolved_output, af::dim4 stride,
                           af::dim4 padding, af::dim4 dilation);

template<typename T>
Array<T> conv2FilterGradient(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<T> &original_filter,
                             const Array<T> &convolved_output, af::dim4 stride,
                             af::dim4 padding, af::dim4 dilation);
}  // namespace cpu
