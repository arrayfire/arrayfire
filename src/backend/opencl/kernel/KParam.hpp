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

#ifndef __OPENCL_VERSION__
// Only define dim_t in host code. dim_t is defined when setting the program
// options in program.cpp
#include <af/defines.h>
#endif

// Defines the size and shape of the data in the OpenCL buffer
typedef struct {
    dim_t dims[4];
    dim_t strides[4];
    dim_t offset;

#ifndef __OPENCL_VERSION__
    dim_t *dims_ptr() { return dims; }
    dim_t *strides_ptr() { return strides; }
#endif

} KParam;

#endif
