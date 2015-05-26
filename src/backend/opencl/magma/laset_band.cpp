/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if 0  // Needs to be enabled when unmqr2 is enabled
#include "magma_data.h"
#include "kernel/laset_band.hpp"

#include <algorithm>

template<typename T> void
magmablas_laset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
                     T offdiag, T diag,
                     cl_mem dA, size_t dA_offset, magma_int_t ldda,
                     magma_queue_t queue)
{
    magma_int_t info = 0;

    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 || k > 1024 )
        info = -4;
    else if ( ldda < std::max(1,m) )
        info = -6;

    if (info != 0) {
        //magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( m == 0 || n == 0 ) {
        return;
    }


    switch (uplo) {
    case MagmaLower: return opencl::kernel::laset_band<T, 0>(m, n, k,
                                                             offdiag, diag, dA, dA_offset, ldda);
    case MagmaUpper: return opencl::kernel::laset_band<T, 1>(m, n, k,
                                                             offdiag, diag, dA, dA_offset, ldda);
    default: return;
    }

}

#define INSTANTIATE(T)                                  \
    template void magmablas_laset_band<T>(              \
        magma_uplo_t uplo,                              \
        magma_int_t m, magma_int_t n, magma_int_t k,    \
        T offdiag, T diag,                              \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,  \
        magma_queue_t queue);                           \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
#endif
