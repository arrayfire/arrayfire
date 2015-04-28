/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __MAGMA_H
#define __MAGMA_H

#include "magma_common.h"

template<typename Ty>
magma_int_t magma_getrf_gpu(magma_int_t m, magma_int_t n,
                            cl_mem dA, size_t dA_offset, magma_int_t ldda,
                            magma_int_t *ipiv,
                            magma_queue_t queue,
                            magma_int_t *info );

#endif
