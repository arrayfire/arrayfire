/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

typedef enum {
    CONVOLVE_UNSUPPORTED_BATCH_MODE = -1, /* invalid inputs */
    ONE2ONE,            /* one signal, one filter   */
    MANY2ONE,           /* many signal, one filter  */
    MANY2MANY,          /* many signal, many filter */
    ONE2MANY            /* one signal, many filter  */
} ConvolveBatchKind;
