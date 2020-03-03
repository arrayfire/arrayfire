/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

// TODO AF_BATCH_UNSUPPORTED is not required and shouldn't happen
//      Code changes are required to handle all cases properly
//      and this enum value should be removed.
typedef enum {
    AF_BATCH_UNSUPPORTED = -1, /* invalid inputs */
    AF_BATCH_NONE,             /* one signal, one filter   */
    AF_BATCH_LHS,              /* many signal, one filter  */
    AF_BATCH_RHS,              /* one signal, many filter  */
    AF_BATCH_SAME,             /* signal and filter have same batch size */
    AF_BATCH_DIFF,             /* signal and filter have different batch size */
} AF_BATCH_KIND;
