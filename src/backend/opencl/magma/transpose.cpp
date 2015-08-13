/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/***********************************************************************
 * Based on MAGMA library http://icl.cs.utk.edu/magma/
 * Below is the original copyright.
 *
 *   -- MAGMA (version 0.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date
 *
 *      @precisions normal z -> s d c
 *
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright 2009-2013
 *
 * Redistribution  and  use  in  source and binary forms, with or without
 * modification,  are  permitted  provided  that the following conditions
 * are met:
 *
 * * Redistributions  of  source  code  must  retain  the above copyright
 *   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * * Redistributions  in  binary  form must reproduce the above copyright
 *   notice,  this list of conditions and the following disclaimer in the
 *   documentation  and/or other materials provided with the distribution.
 * * Neither  the  name of the University of Tennessee, Knoxville nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************/

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
