/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "magma_data.h"
#include "kernel/transpose.hpp"

template<typename T> void
magmablas_transpose(
    magma_int_t m, magma_int_t n,
    cl_mem dA,  size_t dA_offset,  magma_int_t ldda,
    cl_mem dAT, size_t dAT_offset, magma_int_t lddat,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -4;
    else if ( lddat < n )
        info = -6;

    if ( info != 0 ) {
        //magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    int idims[] = {m, n, 1, 1};
    int odims[] = {n, m, 1, 1};
    int istrides[] = {1, ldda, ldda * n, ldda * n};
    int ostrides[] = {1, lddat, lddat * m, lddat * m};

    using namespace opencl;

    if (m % 32 == 0 && n % 32 == 0) {
        kernel::transpose<T, false, true >(makeParam(dAT, dAT_offset, odims, ostrides),
                                           makeParam(dA , dA_offset , idims, istrides));
    } else {
        kernel::transpose<T, false, false>(makeParam(dAT, dAT_offset, odims, ostrides),
                                           makeParam(dA , dA_offset , idims, istrides));
    }
}

#define INSTANTIATE(T)                                      \
    template void magmablas_transpose<T>(                   \
        magma_int_t m, magma_int_t n,                       \
        cl_mem dA,  size_t dA_offset,  magma_int_t ldda,    \
        cl_mem dAT, size_t dAT_offset, magma_int_t lddat,   \
        magma_queue_t queue);                               \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
