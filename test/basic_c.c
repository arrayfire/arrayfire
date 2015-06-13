/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>

int main() {
    af_array out = 0;
    dim_t s[] = {10, 10, 1, 1};
    af_err e = af_randu(&out, 4, s, f32);
    return (AF_SUCCESS != e);
}
