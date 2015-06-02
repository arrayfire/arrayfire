/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "magma_data.h"
#include "kernel/laset.hpp"

#include <algorithm>

template<typename T> void
magmablas_laset(magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                T offdiag, T diag,
                cl_mem dA, size_t dA_offset, magma_int_t ldda,
                magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < std::max(1,m) )
        info = -7;

    if (info != 0) {
        return;  //info;
    }

    if ( m == 0 || n == 0 ) {
        return;
    }


    switch (uplo) {
    case MagmaFull : return opencl::kernel::laset<T, 0>(m, n, offdiag, diag, dA, dA_offset, ldda);
    case MagmaLower: return opencl::kernel::laset<T, 1>(m, n, offdiag, diag, dA, dA_offset, ldda);
    case MagmaUpper: return opencl::kernel::laset<T, 2>(m, n, offdiag, diag, dA, dA_offset, ldda);
    default: return;
    }

}

#define INSTANTIATE(T)                                      \
    template void magmablas_laset<T>(                       \
        magma_uplo_t uplo, magma_int_t m, magma_int_t n,    \
        T offdiag, T diag,                                  \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,      \
        magma_queue_t queue);                               \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
