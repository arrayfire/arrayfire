/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined (WITH_GRAPHICS)
#include <af/array.h>
#include <forge.h>

#ifdef __cplusplus
#include <utility>
namespace af
{
    AFAPI void drawImage(const array &in, const fg_image_handle image);

    AFAPI void drawPlot(const array &X, const array &Y, const fg_plot_handle plot);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    AFAPI af_err af_draw_image(const af_array in, const fg_image_handle image);

    AFAPI af_err af_draw_plot(const af_array X, const af_array Y, const fg_plot_handle plot);
#ifdef __cplusplus
}
#endif

#endif
