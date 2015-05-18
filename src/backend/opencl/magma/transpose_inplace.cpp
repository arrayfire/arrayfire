/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "magma_data.h"
#include "kernel/transpose_inplace.hpp"

template<typename T> void
magmablas_transpose_inplace(
    magma_int_t n,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( ldda < n )
        info = -3;

    if ( info != 0 ) {
        //magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if (n == 0) return;

    int dims[] = {n, n, 1, 1};
    int strides[] = {1, ldda, ldda * n, ldda * n};

    using namespace opencl;

    if (n % 32 == 0) {
        kernel::transpose_inplace<T, false, true >(makeParam(dA , dA_offset , dims, strides));
    } else {
        kernel::transpose_inplace<T, false, false>(makeParam(dA , dA_offset , dims, strides));
    }
}

#define INSTANTIATE(T)                                  \
    template void magmablas_transpose_inplace<T>(       \
        magma_int_t n,                                  \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,  \
        magma_queue_t queue);                           \


INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
