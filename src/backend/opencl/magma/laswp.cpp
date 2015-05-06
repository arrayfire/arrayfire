/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "magma_data.h"
#include "kernel/laswp.hpp"

#include <algorithm>

template<typename T> void
magmablas_laswp(
    magma_int_t n,
    cl_mem dAT, size_t dAT_offset, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 1 )
        info = -4;
    else if ( k2 < 1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        //magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    opencl::kernel::laswp<T>(n, dAT, dAT_offset, ldda, k1, k2, ipiv, inci);
}


#define INSTANTIATE(T)                                      \
    template void magmablas_laswp<T>(                       \
        magma_int_t n,                                      \
        cl_mem dAT, size_t dAT_offset, magma_int_t ldda,    \
        magma_int_t k1, magma_int_t k2,                     \
        const magma_int_t *ipiv, magma_int_t inci,          \
        magma_queue_t queue);                               \


INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
