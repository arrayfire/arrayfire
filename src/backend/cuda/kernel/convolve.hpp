/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include "shared.hpp"

namespace cuda
{

namespace kernel
{

template<typename T, typename accType, int baseDim, bool expand>
void convolve_nd(Param<T> out, CParam<T> signal, CParam<accType> filter, AF_BATCH_KIND kind);

template<typename T, typename accType, int conv_dim, bool expand>
void convolve2(Param<T> out, CParam<T> signal, CParam<accType> filter);

}

}
