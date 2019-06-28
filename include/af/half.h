/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

typedef struct {
    union {
        unsigned short data_ : 16;
        struct {
            unsigned short fraction : 10;
            unsigned short exponent : 5;
            unsigned short sign : 1;
        };
    };
} af_half;

#ifdef __cplusplus
namespace af {
#endif
typedef af_half half;
#ifdef __cplusplus
}
#endif
