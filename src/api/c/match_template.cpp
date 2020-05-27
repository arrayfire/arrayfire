/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <match_template.hpp>
#include <types.hpp>
#include <af/defines.h>
#include <af/vision.h>

#include <type_traits>

using af::dim4;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::conditional;
using std::is_same;

template<typename InType>
static af_array match_template(const af_array& sImg, const af_array tImg,
                               af_match_type mType) {
    using OutType = typename conditional<is_same<InType, double>::value, double,
                                         float>::type;
    return getHandle(match_template<InType, OutType>(
        getArray<InType>(sImg), getArray<InType>(tImg), mType));
}

af_err af_match_template(af_array* out, const af_array search_img,
                         const af_array template_img,
                         const af_match_type m_type) {
    try {
        ARG_ASSERT(3, (m_type >= AF_SAD && m_type <= AF_LSSD));

        const ArrayInfo& sInfo = getInfo(search_img);
        const ArrayInfo& tInfo = getInfo(template_img);

        dim4 const& sDims = sInfo.dims();
        dim4 const& tDims = tInfo.dims();

        dim_t sNumDims = sDims.ndims();
        dim_t tNumDims = tDims.ndims();
        ARG_ASSERT(1, (sNumDims >= 2));
        ARG_ASSERT(2, (tNumDims == 2));

        af_dtype sType = sInfo.getType();
        ARG_ASSERT(1, (sType == tInfo.getType()));

        af_array output = 0;
        switch (sType) {
            case f64:
                output =
                    match_template<double>(search_img, template_img, m_type);
                break;
            case f32:
                output =
                    match_template<float>(search_img, template_img, m_type);
                break;
            case s32:
                output = match_template<int>(search_img, template_img, m_type);
                break;
            case u32:
                output = match_template<uint>(search_img, template_img, m_type);
                break;
            case s16:
                output =
                    match_template<short>(search_img, template_img, m_type);
                break;
            case u16:
                output =
                    match_template<ushort>(search_img, template_img, m_type);
                break;
            case b8:
                output = match_template<char>(search_img, template_img, m_type);
                break;
            case u8:
                output =
                    match_template<uchar>(search_img, template_img, m_type);
                break;
            default: TYPE_ERROR(1, sType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
