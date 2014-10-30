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
    dim_type dims[4];
    dim_type strides[4];
    dim_type offset;
} KParam;
#endif
