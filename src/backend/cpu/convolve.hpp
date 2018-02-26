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

namespace cpu
{

template<typename T, typename accT, int baseDim, bool expand>
Array<T> convolve(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind);

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter);

}
