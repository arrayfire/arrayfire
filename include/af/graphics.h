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
    AFAPI int image(const array &in, const int wId=-1, const char *title=NULL);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_image(int *windowId, const af_array in, const int wId, const char *title);

#ifdef __cplusplus
}
#endif
