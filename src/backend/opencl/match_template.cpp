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
#include <err_opencl.hpp>
#include <match_template.hpp>
#include <kernel/match_template.hpp>

using af::dim4;

namespace opencl
{

template<typename inType, typename outType, af_match_type mType>
Array<outType> match_template(const Array<inType> &sImg, const Array<inType> &tImg)
{
    Array<outType> out = createEmptyArray<outType>(sImg.dims());

    bool needMean = mType==AF_ZSAD || mType==AF_LSAD ||
                    mType==AF_ZSSD || mType==AF_LSSD ||
                    mType==AF_ZNCC;

    if (needMean)
        kernel::matchTemplate<inType, outType, mType, true >(out, sImg, tImg);
    else
        kernel::matchTemplate<inType, outType, mType, false>(out, sImg, tImg);

    return out;
}

#define INSTANTIATE(in_t, out_t)\
    template Array<out_t> match_template<in_t, out_t, AF_SAD >(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_LSAD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_ZSAD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_SSD >(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_LSSD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_ZSSD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_NCC >(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_ZNCC>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_SHD >(const Array<in_t> &sImg, const Array<in_t> &tImg);

INSTANTIATE(double, double)
INSTANTIATE(float ,  float)
INSTANTIATE(char  ,  float)
INSTANTIATE(int   ,  float)
INSTANTIATE(uint  ,  float)
INSTANTIATE(uchar ,  float)

}
