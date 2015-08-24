/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>

namespace opencl
{

namespace kernel
{

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_SCONV_FILTER_LEN = 31;

template<typename T, typename accType, int conv_dim, bool expand, int fLen>
void convolve2(Param out, const Param signal, const Param filter);

template<typename T, typename accT, dim_t cDim, bool expand>
void conv2Helper(Param out, const Param sig, const Param filt, dim_t f);

}

}
