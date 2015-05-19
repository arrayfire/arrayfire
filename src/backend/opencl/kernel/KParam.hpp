/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __KPARAM_H
#define __KPARAM_H
typedef struct
{
    dim_t dims[4];
    dim_t strides[4];
    dim_t offset;
} KParam;
#endif
