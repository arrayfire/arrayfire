/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/seq.h>

af_seq af_make_seq(double begin, double end, double step) {
    af_seq seq = {begin, end, step};
    return seq;
}

