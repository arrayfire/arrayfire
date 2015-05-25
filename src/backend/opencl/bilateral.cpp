/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <bilateral.hpp>
#include <kernel/bilateral.hpp>

using af::dim4;

namespace opencl
{

template<typename inType, typename outType, bool isColor>
Array<outType> bilateral(const Array<inType> &in, const float &s_sigma, const float &c_sigma)
{
    Array<outType> out       = createEmptyArray<outType>(in.dims());
    kernel::bilateral<inType, outType, isColor>(out, in, s_sigma, c_sigma);
    return out;
}

#define INSTANTIATE(inT, outT)\
template Array<outT> bilateral<inT, outT,true >(const Array<inT> &in, const float &s_sigma, const float &c_sigma);\
template Array<outT> bilateral<inT, outT,false>(const Array<inT> &in, const float &s_sigma, const float &c_sigma);

INSTANTIATE(double, double)
INSTANTIATE(float ,  float)
INSTANTIATE(char  ,  float)
INSTANTIATE(int   ,  float)
INSTANTIATE(uint  ,  float)
INSTANTIATE(uchar ,  float)

}
