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

#include <algorithm>

template<typename Ty>
magma_int_t magma_getrf_gpu(
    magma_int_t m, magma_int_t n,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_queue_t queue,
    magma_int_t *info)
{
/*  -- clMAGMA (version 0.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    @date

    Purpose
    =======
    GETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
    A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    =========
    M       (input) INTEGER
    The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
    The number of columns of the matrix A.  N >= 0.

    A       (input/output) an array on the GPU, dimension (LDDA,N).
    On entry, the M-by-N matrix to be factored.
    On exit, the factors L and U from the factorization
    A = P*L*U; the unit diagonal elements of L are not stored.

    LDDA     (input) INTEGER
    The leading dimension of the array A.  LDDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
    The pivot indices; for 1 <= i <= min(M,N), row i of the
    matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
    = 0:  successful exit
    < 0:  if INFO = -i, the i-th argument had an illegal value
    or another error occured, such as memory allocation failed.
    > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
    has been completed, but the factor U is exactly
    singular, and division by zero will occur if it is used
    to solve a system of equations.
    =====================================================================    */

#define  dA(i_, j_) dA,   dA_offset  + (i_)*nb       + (j_)*nb*ldda
#define dAT(i_, j_) dAT,  dAT_offset + (i_)*nb*lddat + (j_)*nb
#define dAP(i_, j_) dAP,               (i_)          + (j_)*maxm
#define work(i_)   (work + (i_))

    static const Ty c_one     = magma_one<Ty>();
    static const Ty c_neg_one = magma_neg_one<Ty>();

    magma_int_t iinfo = 0, nb;
    magma_int_t maxm, maxn, mindim;
    magma_int_t i, j, rows, s, lddat, ldwork;
    cl_mem dAT, dAP;
    Ty *work;
    size_t dAT_offset;

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < std::max(1,m))
        *info = -4;

    if (*info != 0) {
        //magma_xerbla(__func__, -(*info));
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    gemm_func<Ty> gpu_gemm;
    trsm_func<Ty> gpu_trsm;
    getrf_func<Ty> cpu_getrf;

    /* Function Body */
    mindim = std::min(m, n);
    nb     = magma_get_getrf_nb<Ty>(m);
    s      = mindim / nb;

    if (nb <= 1 || nb >= std::min(m,n)) {
        /* Use CPU code. */
        if (MAGMA_SUCCESS != magma_malloc_cpu<Ty>(&work, m*n)) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_getmatrix<Ty>(m, n, dA(0,0), ldda, work(0), m, queue);
        cpu_getrf(LAPACK_COL_MAJOR, m, n, work, m, ipiv);
        magma_setmatrix<Ty>(m, n, work(0), m, dA(0,0), ldda, queue);
        magma_free_cpu(work);
    }
    else {
        /* Use hybrid blocked code. */
        maxm = ((m + 31)/32)*32;
        maxn = ((n + 31)/32)*32;

        if (MAGMA_SUCCESS != magma_malloc<Ty>(&dAP, nb*maxm)) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        // square matrices can be done in place;
        // rectangular requires copy to transpose
        if (m == n) {
            dAT = dA;
            dAT_offset = dA_offset;
            lddat = ldda;
            magmablas_transpose_inplace<Ty>(m, dAT(0,0), lddat, queue);
        }
        else {
            lddat = maxn;  // N-by-M
            dAT_offset = 0;
            if (MAGMA_SUCCESS != magma_malloc<Ty>(&dAT, lddat*maxm)) {
                magma_free(dAP);
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            magmablas_transpose<Ty>(m, n, dA(0,0), ldda, dAT(0,0), lddat, queue);
        }

        ldwork = maxm;
        if (MAGMA_SUCCESS != magma_malloc_cpu<Ty>(&work, ldwork*nb)) {
            magma_free(dAP);
            if (dA != dAT)
                magma_free(dAT);

            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }

        cl_event event = 0;


        for(j=0; j < s; j++) {

            // download j-th panel
            magmablas_transpose<Ty>(nb, m-j*nb, dAT(j,j), lddat, dAP(0,0), maxm, queue);

            magma_getmatrix<Ty>(m-j*nb, nb, dAP(0,0), maxm, work(0), ldwork, queue);

            if (j > 0 && n > (j + 1) * nb) {
                gpu_trsm(clblasColumnMajor,
                         clblasRight, clblasUpper, clblasNoTrans, clblasUnit,
                         n - (j+1)*nb, nb,
                         c_one,
                         dAT(j-1,j-1), lddat,
                         dAT(j-1,j+1), lddat,
                         1, &queue, 0, nullptr, &event);

                if (m > j * nb)  {
                    gpu_gemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                         n-(j+1)*nb, m-j*nb, nb,
                         c_neg_one,
                         dAT(j-1,j+1), lddat,
                         dAT(j,  j-1), lddat,
                         c_one,
                         dAT(j,  j+1), lddat,
                         1, &queue, 0, nullptr, &event);
                }
            }

            // do the cpu part
            rows = m - j*nb;
            cpu_getrf(LAPACK_COL_MAJOR, rows, nb, work, ldwork, ipiv+j*nb);
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j*nb;

            for(i=j*nb; i < j*nb + nb; ++i) {
                ipiv[i] += j*nb;
            }
            magmablas_laswp<Ty>(n, dAT(0,0), lddat, j*nb + 1, j*nb + nb, ipiv, 1, queue);

            // upload j-th panel
            magma_setmatrix<Ty>(m-j*nb, nb, work(0), ldwork, dAP(0,0), maxm, queue);

            magmablas_transpose<Ty>(m-j*nb, nb, dAP(0,0), maxm, dAT(j,j), lddat, queue);

            // do the small non-parallel computations (next panel update)
            if (s > (j+1)) {
                gpu_trsm(clblasColumnMajor,
                         clblasRight, clblasUpper, clblasNoTrans, clblasUnit,
                         nb, nb,
                         c_one,
                         dAT(j, j  ), lddat,
                         dAT(j, j+1), lddat,
                         1, &queue, 0, nullptr, &event);


                gpu_gemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                         nb, m-(j+1)*nb, nb,
                         c_neg_one,
                         dAT(j,   j+1), lddat,
                         dAT(j+1, j  ), lddat,
                         c_one,
                         dAT(j+1, j+1), lddat,
                         1, &queue, 0, nullptr, &event);
            }
            else {
                if (n > s * nb) {
                    gpu_trsm(clblasColumnMajor,
                             clblasRight, clblasUpper, clblasNoTrans, clblasUnit,
                             n-s*nb, nb,
                             c_one,
                             dAT(j, j  ), lddat,
                             dAT(j, j+1), lddat,
                             1, &queue, 0, nullptr, &event);
                }

                if ((n > (j+1) * nb) && (m > (j+1) * nb)) {
                    gpu_gemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                             n-(j+1)*nb, m-(j+1)*nb, nb,
                             c_neg_one,
                             dAT(j,   j+1), lddat,
                             dAT(j+1, j  ), lddat,
                             c_one,
                             dAT(j+1, j+1), lddat,
                             1, &queue, 0, nullptr, &event);
                }
            }
        }

        magma_int_t nb0 = std::min(m - s*nb, n - s*nb);

        if (nb0 > 0 && m > s * nb) {
            rows = m - s*nb;

            magmablas_transpose<Ty>(nb0, rows, dAT(s,s), lddat, dAP(0,0), maxm, queue);
            magma_getmatrix<Ty>(rows, nb0, dAP(0,0), maxm, work(0), ldwork, queue);

            // do the cpu part
            cpu_getrf(LAPACK_COL_MAJOR, rows, nb0, work, ldwork, ipiv+s*nb);
            if (*info == 0 && iinfo > 0)
                *info = iinfo + s*nb;

            for(i=s*nb; i < s*nb + nb0; ++i) {
                ipiv[i] += s*nb;
            }
            magmablas_laswp<Ty>(n, dAT(0,0), lddat, s*nb + 1, s*nb + nb0, ipiv, 1, queue);

            // upload j-th panel
            magma_setmatrix<Ty>(rows, nb0, work(0), ldwork, dAP(0,0), maxm, queue);
            magmablas_transpose<Ty>(rows, nb0, dAP(0,0), maxm, dAT(s,s), lddat, queue);

            if (n > s * nb + nb0) {
                gpu_trsm(clblasColumnMajor,
                         clblasRight, clblasUpper, clblasNoTrans, clblasUnit,
                         n-s*nb-nb0, nb0,
                         c_one, dAT(s,s),     lddat,
                         dAT(s,s)+nb0, lddat, 1, &queue, 0, nullptr, &event);
            }
        }

        // undo transpose
        if (dA == dAT) {
            magmablas_transpose_inplace<Ty>(m, dAT(0,0), lddat, queue);
        }
        else {
            magmablas_transpose<Ty>(n, m, dAT(0,0), lddat, dA(0,0), ldda, queue);
            magma_free(dAT);
        }

        magma_free(dAP);
        magma_free_cpu(work);
    }

    return *info;
} /* getrf_gpu */

#undef dAT

#define INSTANTIATE(T)                                  \
    template magma_int_t magma_getrf_gpu<T>(            \
        magma_int_t m, magma_int_t n,                   \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,  \
        magma_int_t *ipiv,                              \
        magma_queue_t queue,                            \
        magma_int_t *info);                             \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
