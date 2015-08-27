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

#include <algorithm>

template<typename Ty>
magma_int_t magma_potrf_gpu(
    magma_uplo_t   uplo, magma_int_t    n,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t*   info)
{
/*  -- clMAGMA (version 0.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    @date

    Purpose
    =======
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
    dA = U\*\*H * U,  if UPLO = 'U', or
    dA = L  * L\*\*H,  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    =========
    UPLO    (input) INTEGER
    = MagmaUpper:  Upper triangle of dA is stored;
    = MagmaLower:  Lower triangle of dA is stored.

    N       (input) INTEGER
    The order of the matrix dA.  N >= 0.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
    On entry, the Hermitian matrix dA.  If UPLO = 'U', the leading
    N-by-N upper triangular part of dA contains the upper
    triangular part of the matrix dA, and the strictly lower
    triangular part of dA is not referenced.  If UPLO = 'L', the
    leading N-by-N lower triangular part of dA contains the lower
    triangular part of the matrix dA, and the strictly upper
    triangular part of dA is not referenced.

    On exit, if INFO = 0, the factor U or L from the Cholesky
    factorization dA = U\*\*H*U or dA = L*L\*\*H.

    LDDA    (input) INTEGER
    The leading dimension of the array dA.  LDDA >= max(1,N).
    To benefit from coalescent memory accesses LDDA must be
    divisible by 16.

    INFO    (output) INTEGER
    = 0:  successful exit
    < 0:  if INFO = -i, the i-th argument had an illegal value
    > 0:  if INFO = i, the leading minor of order i is not
    positive definite, and the factorization could not be
    completed.
    =====================================================================   */

// produces pointer and offset as two args to magmaBLAS routines
#define dA(i,j)  dA, ((dA_offset) + (i) + (j)*ldda)

// produces pointer as single arg to BLAS routines
#define A(i,j)  &A[ (i) + (j)*lda ]

    magma_int_t j, jb, nb;
    static const Ty  z_one = magma_one<Ty>();
    static const Ty mz_one = magma_neg_one<Ty>();
    static const double    one =  1.0;
    static const double  m_one = -1.0;

    static const clblasTranspose transType = magma_is_real<Ty>() ? clblasTrans : clblasConjTrans;

    Ty* work;
    magma_int_t err;

    *info = 0;
    if (uplo != MagmaUpper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < std::max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        //magma_xerbla(__func__, -(*info));
        return *info;
    }

    nb = magma_get_potrf_nb<Ty>(n);

    gpu_blas_gemm_func<Ty> gpu_blas_gemm;
    gpu_blas_trsm_func<Ty> gpu_blas_trsm;
    gpu_blas_herk_func<Ty> gpu_blas_herk;
    cpu_lapack_potrf_func<Ty> cpu_lapack_potrf;


    err = magma_malloc_cpu<Ty>( &work, nb*nb);
    if (err != MAGMA_SUCCESS) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    magma_event_t event = NULL;
    cl_event blas_event = NULL;

    if ((nb <= 1) || (nb >= n)) {
        // use unblocked code
        magma_getmatrix<Ty>(n, n, dA, dA_offset, ldda, work, n, queue);

        LAPACKE_CHECK(cpu_lapack_potrf(
                          uplo == MagmaUpper ? *MagmaUpperStr : *MagmaLowerStr,
                          n, work, n));

        magma_setmatrix<Ty>(n, n, work, n, dA, dA_offset, ldda, queue);
    }
    else {
        if (uplo == MagmaUpper) {
            // --------------------
            // compute Cholesky factorization A = U'*U
            // using the left looking algorithm
            for(j = 0; j < n; j += nb) {
                // apply all previous updates to diagonal block
                jb = std::min(nb, n-j);
                if (j > 0) {
                    CLBLAS_CHECK(gpu_blas_herk(
                                     clblasUpper, transType,
                                     jb, j,
                                     m_one,
                                     dA(0,j), ldda,
                                     one,
                                     dA(j,j), ldda,
                                     1, &queue, 0, nullptr, &blas_event));
                }

                // start asynchronous data transfer
                magma_getmatrix_async<Ty>(jb, jb, dA(j,j), ldda, work, jb, queue, &event);

                // apply all previous updates to block row right of diagonal block
                if (j+jb < n) {
                    CLBLAS_CHECK(gpu_blas_gemm(
                                     transType, clblasNoTrans,
                                     jb, n-j-jb, j,
                                     mz_one,
                                     dA(0, j   ), ldda,
                                     dA(0, j+jb), ldda,
                                     z_one,
                                     dA(j, j+jb), ldda,
                                     1, &queue, 0, nullptr, &blas_event));
                }

                // simultaneous with above zgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                magma_event_sync(event);
                LAPACKE_CHECK(cpu_lapack_potrf( *MagmaUpperStr, jb, work, jb));

                if (*info != 0) {
                    assert(*info > 0);
                    *info += j;
                    break;
                }

                magma_setmatrix_async<Ty>(jb, jb, work, jb, dA(j,j), ldda, queue, &event);

                // apply diagonal block to block row right of diagonal block
                if (j+jb < n) {
                    magma_event_sync(event);
                    CLBLAS_CHECK(gpu_blas_trsm(
                                     clblasLeft, clblasUpper,
                                     transType, clblasNonUnit,
                                     jb, n-j-jb,
                                     z_one,
                                     dA(j, j   ), ldda,
                                     dA(j, j+jb), ldda,
                                     1, &queue, 0, nullptr, &blas_event));
                }
            }
        }
        else {
            // --------------------
            // compute Cholesky factorization A = L*L'
            // using the left looking algorithm
            for(j = 0; j < n; j += nb) {
                // apply all previous updates to diagonal block
                jb = std::min(nb, n-j);
                if (j>0) {
                    CLBLAS_CHECK(gpu_blas_herk(
                                     clblasLower, clblasNoTrans, jb, j,
                                     m_one,
                                     dA(j, 0), ldda,
                                     one,
                                     dA(j, j), ldda,
                                     1, &queue, 0, nullptr, &blas_event));
                }

                // start asynchronous data transfer
                magma_getmatrix_async<Ty>(jb, jb, dA(j,j), ldda, work, jb, queue, &event);

                // apply all previous updates to block column below diagonal block
                if (j+jb < n) {
                    CLBLAS_CHECK(gpu_blas_gemm(
                                     clblasNoTrans, transType,
                                     n-j-jb, jb, j,
                                     mz_one,
                                     dA(j+jb, 0), ldda,
                                     dA(j,    0), ldda,
                                     z_one,
                                     dA(j+jb, j), ldda,
                                     1, &queue, 0, nullptr, &blas_event));
                }

                // simultaneous with above zgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                magma_event_sync(event);
                LAPACKE_CHECK(cpu_lapack_potrf(
                                  *MagmaLowerStr, jb, work, jb));
                if (*info != 0) {
                    assert(*info > 0);
                    *info += j;
                    break;
                }
                magma_setmatrix_async<Ty>(jb, jb, work, jb, dA(j,j), ldda, queue, &event);

                // apply diagonal block to block column below diagonal
                if (j+jb < n) {
                    magma_event_sync(event);
                    CLBLAS_CHECK(gpu_blas_trsm(
                                     clblasRight, clblasLower, transType, clblasNonUnit,
                                     n-j-jb, jb,
                                     z_one,
                                     dA(j   , j), ldda,
                                     dA(j+jb, j), ldda,
                                     1, &queue, 0, nullptr, &blas_event));
                }
            }
        }
    }

    magma_queue_sync(queue);
    magma_free_cpu(work);

    return *info;
}

#define INSTANTIATE(T)                                  \
    template magma_int_t magma_potrf_gpu<T>(            \
        magma_uplo_t   uplo, magma_int_t    n,          \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,  \
        magma_queue_t queue,                            \
        magma_int_t*   info);                           \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
