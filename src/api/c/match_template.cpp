/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/vision.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <match_template.hpp>

using af::dim4;
using namespace detail;

template<typename inType, typename outType>
static
af_array match_template(const af_array &sImg, const af_array tImg, af_match_type mType)
{
    switch(mType) {
        case AF_SAD : return getHandle(match_template<inType, outType, AF_SAD >(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_ZSAD: return getHandle(match_template<inType, outType, AF_ZSAD>(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_LSAD: return getHandle(match_template<inType, outType, AF_LSAD>(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_SSD : return getHandle(match_template<inType, outType, AF_SSD >(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_ZSSD: return getHandle(match_template<inType, outType, AF_ZSSD>(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_LSSD: return getHandle(match_template<inType, outType, AF_LSSD>(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_NCC : return getHandle(match_template<inType, outType, AF_NCC >(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_ZNCC: return getHandle(match_template<inType, outType, AF_ZNCC>(getArray<inType>(sImg), getArray<inType>(tImg)));
        case AF_SHD : return getHandle(match_template<inType, outType, AF_SHD >(getArray<inType>(sImg), getArray<inType>(tImg)));
        default:      return getHandle(match_template<inType, outType, AF_SAD >(getArray<inType>(sImg), getArray<inType>(tImg)));
    }
}

af_err af_match_template(af_array *out, const af_array search_img, const af_array template_img, const af_match_type m_type)
{
    try {
        ARG_ASSERT(3, (m_type>=AF_SAD && m_type<=AF_LSSD));

        ArrayInfo sInfo = getInfo(search_img);
        ArrayInfo tInfo = getInfo(template_img);

        dim4 const sDims = sInfo.dims();
        dim4 const tDims = tInfo.dims();

        dim_t sNumDims= sDims.ndims();
        dim_t tNumDims= tDims.ndims();
        ARG_ASSERT(1, (sNumDims>=2));
        ARG_ASSERT(2, (tNumDims==2));

        af_dtype sType = sInfo.getType();
        ARG_ASSERT(1, (sType==tInfo.getType()));

        af_array output = 0;
        switch(sType) {
            case f64: output = match_template<double, double>(search_img, template_img, m_type); break;
            case f32: output = match_template<float ,  float>(search_img, template_img, m_type); break;
            case s32: output = match_template<int   ,  float>(search_img, template_img, m_type); break;
            case u32: output = match_template<uint  ,  float>(search_img, template_img, m_type); break;
            case  b8: output = match_template<char  ,  float>(search_img, template_img, m_type); break;
            case  u8: output = match_template<uchar ,  float>(search_img, template_img, m_type); break;
            default : TYPE_ERROR(1, sType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
