/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <convolve_common.hpp>

namespace cuda
{

template<typename T, typename accT, dim_type baseDim, bool expand>
Array<T> * convolve(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);

template<typename T, typename accT, bool expand>
Array<T> * convolve2(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);

}
