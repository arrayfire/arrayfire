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

namespace arrayfire {
namespace cuda {
template<typename Ti, typename Tw, typename To>
To mean(const Array<Ti>& in);

template<typename T, typename Tw>
T mean(const Array<T>& in, const Array<Tw>& wts);

template<typename Ti, typename Tw, typename To>
Array<To> mean(const Array<Ti>& in, const int dim);

template<typename T, typename Tw>
Array<T> mean(const Array<T>& in, const Array<Tw>& wts, const int dim);

}  // namespace cuda
}  // namespace arrayfire
