/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/array.h>
#include <af/image.h>
#include <err_common.hpp>

template<af_cspace_t FROM, af_cspace_t TO>
af_err convert(af_array *out, const af_array image)
{
    return AF_ERR_NOT_SUPPORTED;
}

#define INSTANTIATE_CSPACE_DEFS1(F, T, FUNC)                \
template<>                                                  \
af_err convert<F, T>(af_array *out, const af_array image)   \
{                                                           \
    return FUNC(out, image);                                \
}

#define INSTANTIATE_CSPACE_DEFS2(F, T, FUNC, ...)           \
template<>                                                  \
af_err convert<F, T>(af_array *out, const af_array image)   \
{                                                           \
    return FUNC(out, image, __VA_ARGS__);                   \
}

INSTANTIATE_CSPACE_DEFS1(AF_HSV  , AF_RGB  , af_hsv2rgb  );
INSTANTIATE_CSPACE_DEFS1(AF_RGB  , AF_HSV  , af_rgb2hsv  );
INSTANTIATE_CSPACE_DEFS2(AF_RGB  , AF_GRAY , af_rgb2gray , 0.2126f, 0.7152f, 0.0722f);
INSTANTIATE_CSPACE_DEFS2(AF_GRAY , AF_RGB  , af_gray2rgb , 1.0f, 1.0f, 1.0f);
INSTANTIATE_CSPACE_DEFS2(AF_YCbCr, AF_RGB  , af_ycbcr2rgb, AF_YCC_601);
INSTANTIATE_CSPACE_DEFS2(AF_RGB  , AF_YCbCr, af_rgb2ycbcr, AF_YCC_601);

template<af_cspace_t FROM>
static af_err convert(af_array *out, const af_array image, const af_cspace_t to)
{
    switch(to) {
        case AF_GRAY : return convert<FROM, AF_GRAY >(out, image);
        case AF_RGB  : return convert<FROM, AF_RGB  >(out, image);
        case AF_HSV  : return convert<FROM, AF_HSV  >(out, image);
        case AF_YCbCr: return convert<FROM, AF_YCbCr>(out, image);
        default: return AF_ERR_ARG;
    }
}

af_err af_color_space(af_array *out, const af_array image, const af_cspace_t to, const af_cspace_t from)
{
    if (from==to) {
        return af_retain_array(out, image);
    }

    switch(from) {
        case AF_GRAY : return convert<AF_GRAY >(out, image, to);
        case AF_RGB  : return convert<AF_RGB  >(out, image, to);
        case AF_HSV  : return convert<AF_HSV  >(out, image, to);
        case AF_YCbCr: return convert<AF_YCbCr>(out, image, to);
        default: return AF_ERR_ARG;
    }
}
