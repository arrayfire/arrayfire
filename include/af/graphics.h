/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>

#ifdef __cplusplus
#include <utility>
namespace af
{
    AFAPI int image(const array &in, const int wId=-1, const char *title=NULL, const float scale = 1.0f);

    AFAPI int image(const array &in, const float scale_w, const float scale_h,
                    const int wId=-1, const char *title=NULL);

    AFAPI int image(const array &in, const dim_type disp_w, const dim_type disp_h,
                    const int wId=-1, const char *title=NULL);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_image_s(int *windowId, const af_array in, const int wId, const char *title,
                            const float scale_w, const float scale_h);

    AFAPI af_err af_image_d(int *windowId, const af_array in, const int wId, const char *title,
                            const dim_type disp_w, const dim_type disp_h);

#ifdef __cplusplus
}
#endif
