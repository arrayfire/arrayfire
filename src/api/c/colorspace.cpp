/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/image.h>
#include <err_common.hpp>

af_err af_color_space(af_array *out, const af_array image, const af_cspace_t to, const af_cspace_t from)
{
    bool hsv2rgb  = (from==AF_HSV  && to==AF_RGB );
    bool rgb2hsv  = (from==AF_RGB  && to==AF_HSV );
    bool gray2rgb = (from==AF_GRAY && to==AF_RGB );
    bool rgb2gray = (from==AF_RGB  && to==AF_GRAY);

    ARG_ASSERT(2, (hsv2rgb || rgb2hsv || gray2rgb || rgb2gray));

    af_err result = AF_SUCCESS;

    if (hsv2rgb)  result = af_hsv2rgb(out, image);
    if (rgb2hsv)  result = af_rgb2hsv(out, image);
    if (gray2rgb) result = af_gray2rgb(out, image, 1.0f, 1.0f, 1.0f);
    if (rgb2gray) result = af_rgb2gray(out, image, 0.2126f, 0.7152f, 0.0722f);

    return result;
}
