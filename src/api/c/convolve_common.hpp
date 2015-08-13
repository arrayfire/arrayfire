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
    CONVOLVE_BATCH_UNSUPPORTED = -1, /* invalid inputs */
    CONVOLVE_BATCH_NONE,          /* one signal, one filter   */
    CONVOLVE_BATCH_SIGNAL,        /* many signal, one filter  */
    CONVOLVE_BATCH_KERNEL,        /* one signal, many filter  */
    CONVOLVE_BATCH_SAME,          /* signal and filter have same batch size */
    CONVOLVE_BATCH_DIFF,          /* signal and filter have different batch size */
} ConvolveBatchKind;
