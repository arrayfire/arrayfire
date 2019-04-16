/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <af/array.h>
#include <af/defines.h>
#include <af/image.h>

template<af_cspace_t FROM, af_cspace_t TO>
void color_space(af_array *out, const af_array image) {
    UNUSED(out);
    UNUSED(image);
    AF_ERROR(
        "Color Space: Conversion from source type to output type not supported",
        AF_ERR_NOT_SUPPORTED);
}

#define INSTANTIATE_CSPACE_DEFS1(F, T, FUNC)                       \
    template<>                                                     \
    void color_space<F, T>(af_array * out, const af_array image) { \
        AF_CHECK(FUNC(out, image));                                \
    }

#define INSTANTIATE_CSPACE_DEFS2(F, T, FUNC, ...)                  \
    template<>                                                     \
    void color_space<F, T>(af_array * out, const af_array image) { \
        AF_CHECK(FUNC(out, image, __VA_ARGS__));                   \
    }

INSTANTIATE_CSPACE_DEFS1(AF_HSV, AF_RGB, af_hsv2rgb);
INSTANTIATE_CSPACE_DEFS1(AF_RGB, AF_HSV, af_rgb2hsv);

INSTANTIATE_CSPACE_DEFS2(AF_RGB, AF_GRAY, af_rgb2gray, 0.2126f, 0.7152f,
                         0.0722f);
INSTANTIATE_CSPACE_DEFS2(AF_GRAY, AF_RGB, af_gray2rgb, 1.0f, 1.0f, 1.0f);
INSTANTIATE_CSPACE_DEFS2(AF_YCbCr, AF_RGB, af_ycbcr2rgb, AF_YCC_601);
INSTANTIATE_CSPACE_DEFS2(AF_RGB, AF_YCbCr, af_rgb2ycbcr, AF_YCC_601);

template<af_cspace_t FROM>
static void color_space(af_array *out, const af_array image,
                        const af_cspace_t to) {
    switch (to) {
        case AF_GRAY: color_space<FROM, AF_GRAY>(out, image); break;
        case AF_RGB: color_space<FROM, AF_RGB>(out, image); break;
        case AF_HSV: color_space<FROM, AF_HSV>(out, image); break;
        case AF_YCbCr: color_space<FROM, AF_YCbCr>(out, image); break;
        default:
            AF_ERROR("Incorrect enum value for output color type", AF_ERR_ARG);
    }
}

af_err af_color_space(af_array *out, const af_array image, const af_cspace_t to,
                      const af_cspace_t from) {
    try {
        if (from == to) { return af_retain_array(out, image); }

        ARG_ASSERT(2, (to == AF_GRAY || to == AF_RGB || to == AF_HSV ||
                       to == AF_YCbCr));
        ARG_ASSERT(2, (from == AF_GRAY || from == AF_RGB || from == AF_HSV ||
                       from == AF_YCbCr));

        switch (from) {
            case AF_GRAY: color_space<AF_GRAY>(out, image, to); break;
            case AF_RGB: color_space<AF_RGB>(out, image, to); break;
            case AF_HSV: color_space<AF_HSV>(out, image, to); break;
            case AF_YCbCr: color_space<AF_YCbCr>(out, image, to); break;
            default:
                AF_ERROR("Incorrect enum value for input color type",
                         AF_ERR_ARG);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
