/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "magma_data.h"
#include "kernel/swapdblk.hpp"

template<typename T> void
magmablas_swapdblk(magma_int_t n, magma_int_t nb,
                   cl_mem dA, magma_int_t dA_offset, magma_int_t ldda, magma_int_t inca,
                   cl_mem dB, magma_int_t dB_offset, magma_int_t lddb, magma_int_t incb,
                   magma_queue_t queue)
{
    opencl::kernel::swapdblk<T>(n, nb,
                                dA, dA_offset, ldda, inca,
                                dB, dB_offset, lddb, incb);
}


#define INSTANTIATE(T)                                                  \
    template void magmablas_swapdblk<T>(magma_int_t n, magma_int_t nb,  \
                                        cl_mem dA, magma_int_t dA_offset, \
                                        magma_int_t ldda, magma_int_t inca, \
                                        cl_mem dB, magma_int_t dB_offset, \
                                        magma_int_t lddb, magma_int_t incb, \
                                        magma_queue_t queue);           \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
