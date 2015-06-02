/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace cuda
{

template<typename T, af_border_type edge_pad>
Array<T> medfilt(const Array<T> &in, dim_t w_len, dim_t w_wid);

}
