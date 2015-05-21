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

#include "magma.h"
#include "magma_blas.h"
#include "magma_data.h"
#include "magma_cpu_lapack.h"
#include "magma_helper.h"
#include "magma_sync.h"

#include <platform.hpp>
#include <algorithm>
#include <string>

template<typename Ty>  magma_int_t
magma_getrs_gpu(magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                cl_mem dA, size_t dA_offset, magma_int_t ldda,
                magma_int_t *ipiv,
                cl_mem dB, size_t dB_offset, magma_int_t lddb,
                magma_queue_t queue,
                magma_int_t *info)
{
/*  -- clMagma (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    Solves a system of linear equations
      A * X = B  or  A' * X = B
    with a general N-by-N matrix A using the LU factorization computed by ZGETRF_GPU.

    Arguments
    =========
    TRANS   (input) CHARACTER*1
            Specifies the form of the system of equations:
            = 'N':  A * X = B  (No transpose)
            = 'T':  A'* X = B  (Transpose)
            = 'C':  A'* X = B  (Conjugate transpose = Transpose)

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input) COMPLEX_16 array on the GPU, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U as computed
            by ZGETRF_GPU.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from ZGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) COMPLEX_16 array on the GPU, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    HWORK   (workspace) COMPLEX_16 array, dimension N*NRHS
    =====================================================================    */

    static const Ty c_one = magma_one<Ty>();
    Ty *work = NULL;
    int notran = (trans == MagmaNoTrans);
    magma_int_t i1, i2, inc;

    *info = 0;
    if ( (! notran) &&
         (trans != MagmaTrans) &&
         (trans != MagmaConjTrans) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < std::max(1,n)) {
        *info = -5;
    } else if (lddb < std::max(1,n)) {
        *info = -8;
    }
    if (*info != 0) {
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }

    magma_malloc_cpu<Ty>( &work, n*nrhs );
    if ( work == NULL ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    i1 = 1;
    i2 = n;

    laswp_func<Ty> cpu_laswp;
    trsm_func<Ty> gpu_trsm;
    trsv_func<Ty> gpu_trsv;

    cl_event event = NULL;

    clblasTranspose cltrans =(trans == MagmaNoTrans) ? clblasNoTrans :
        (trans == MagmaTrans ? clblasTrans : clblasConjTrans);

    std::string pName = opencl::getPlatformName(opencl::getDevice());
    bool cond = pName.find("NVIDIA") != std::string::npos;
    cl_mem dAT = 0;
    if (nrhs > 1 && cond) {
        magma_malloc<Ty>(&dAT, n * n);
        magmablas_transpose<Ty>(n, n, dA, dA_offset, ldda, dAT, 0, n, queue);
    }
    if (notran) {
        inc = 1;

        /* Solve A * X = B. */
        magma_getmatrix<Ty>( n, nrhs, dB, dB_offset, lddb, work, n, queue );
        cpu_laswp(LAPACK_COL_MAJOR, nrhs, work, n, i1, i2, ipiv, inc);
        magma_setmatrix<Ty>( n, nrhs, work, n, dB, dB_offset, lddb, queue );
        if ( nrhs == 1) {
            gpu_trsv(clblasColumnMajor, clblasLower, clblasNoTrans, clblasUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, 1, &queue, 0, nullptr, &event);
            gpu_trsv(clblasColumnMajor, clblasUpper, clblasNoTrans, clblasNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, 1, &queue, 0, nullptr, &event);
        } else {
            gpu_trsm(clblasColumnMajor, clblasLeft, clblasLower, clblasNoTrans, clblasUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB, dB_offset, lddb, 1, &queue, 0, nullptr, &event);

            if(cond) {
                gpu_trsm(clblasColumnMajor, clblasLeft, clblasLower, clblasTrans, clblasNonUnit, n, nrhs, c_one, dAT, 0, n, dB, dB_offset, lddb, 1, &queue, 0, nullptr, &event);
            } else {
                gpu_trsm(clblasColumnMajor, clblasLeft, clblasUpper, clblasNoTrans, clblasNonUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB, dB_offset, lddb, 1, &queue, 0, nullptr, &event);
            }
        }
    } else {
        inc = -1;

        /* Solve A' * X = B. */
        if ( nrhs == 1) {
            gpu_trsv(clblasColumnMajor, clblasUpper, cltrans, clblasNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, 1, &queue, 0, nullptr, &event);
            gpu_trsv(clblasColumnMajor, clblasLower, cltrans, clblasUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, 1, &queue, 0, nullptr, &event);
        } else {
            if(cond) {
                gpu_trsm(clblasColumnMajor, clblasLeft, clblasLower, clblasNoTrans, clblasNonUnit, n, nrhs, c_one, dAT, 0, n, dB, dB_offset, lddb, 1, &queue, 0, nullptr, &event);
            } else {
                gpu_trsm(clblasColumnMajor, clblasLeft, clblasUpper, cltrans, clblasNonUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB, dB_offset, lddb, 1, &queue, 0, nullptr, &event);
            }
            gpu_trsm(clblasColumnMajor, clblasLeft, clblasLower, cltrans, clblasUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB, dB_offset, lddb, 1, &queue, 0, nullptr, &event);
        }
        magma_getmatrix<Ty>( n, nrhs, dB, dB_offset, lddb, work, n, queue );
        cpu_laswp(LAPACK_COL_MAJOR, nrhs, work, n, i1, i2, ipiv, inc);
        magma_setmatrix<Ty>( n, nrhs, work, n, dB, dB_offset, lddb, queue );
    }

    if (nrhs > 1 && dAT != 0) magma_free(dAT);
    magma_free_cpu(work);
    return *info;
}

#define INSTANTIATE(T)                                                  \
    template  magma_int_t                                               \
    magma_getrs_gpu<T>(magma_trans_t trans, magma_int_t n, magma_int_t nrhs, \
                       cl_mem dA, size_t dA_offset, magma_int_t ldda,   \
                       magma_int_t *ipiv,                               \
                       cl_mem dB, size_t dB_offset, magma_int_t lddb,   \
                       magma_queue_t queue,                             \
                       magma_int_t *info);                              \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
