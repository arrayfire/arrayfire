/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/convolve/conv_common.hpp>

namespace opencl
{

namespace kernel
{

template<typename T, typename aT, bool expand>
void conv3(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt)
{
    convNHelper<T, aT, 3, expand>(p, out, sig, filt);
}

#define INSTANTIATE(T, accT)  \
    template void conv3<T, accT, true >(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \
    template void conv3<T, accT, false>(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

}
