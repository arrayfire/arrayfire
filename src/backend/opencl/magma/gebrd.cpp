/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include "magma_cpu_blas.h"
#include "magma_helper.h"
#include "magma_sync.h"
#include <traits.hpp>
#include <types.hpp>

#include <algorithm>

// produces pointer and offset as two args to magmaBLAS routines
#define dA(i,j)  da, ((da_offset) + (i) + (j)*ldda)
// produces pointer as single arg to BLAS routines
#define A(i,j)  &a[ (i) + (j)*lda ]

template<typename Ty>
magma_int_t
magma_gebrd_hybrid(
    magma_int_t m, magma_int_t n,
    Ty *a, magma_int_t lda,
    cl_mem da, size_t da_offset, magma_int_t ldda,
    void *_d, void *_e,
    Ty *tauq, Ty *taup,
    Ty *work, magma_int_t lwork,
    magma_queue_t queue,
    magma_int_t *info,
    bool copy)
{
/*  -- MAGMA (version 1.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    @date

    Purpose
    =======
    ZGEBRD reduces a general complex M-by-N matrix A to upper or lower
    bidiagonal form B by an orthogonal transformation: Q**H * A * P = B.

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

    Arguments
    =========
    M       (input) INTEGER
    The number of rows in the matrix A.  M >= 0.

    N       (input) INTEGER
    The number of columns in the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
    On entry, the M-by-N general matrix to be reduced.
    On exit,
    if m >= n, the diagonal and the first superdiagonal are
    overwritten with the upper bidiagonal matrix B; the
    elements below the diagonal, with the array TAUQ, represent
    the orthogonal matrix Q as a product of elementary
    reflectors, and the elements above the first superdiagonal,
    with the array TAUP, represent the orthogonal matrix P as
    a product of elementary reflectors;
    if m < n, the diagonal and the first subdiagonal are
    overwritten with the lower bidiagonal matrix B; the
    elements below the first subdiagonal, with the array TAUQ,
    represent the orthogonal matrix Q as a product of
    elementary reflectors, and the elements above the diagonal,
    with the array TAUP, represent the orthogonal matrix P as
    a product of elementary reflectors.
    See Further Details.

    LDA     (input) INTEGER
    The leading dimension of the array A.  LDA >= max(1,M).

    D       (output) double precision array, dimension (min(M,N))
    The diagonal elements of the bidiagonal matrix B:
    D(i) = A(i,i).

    E       (output) double precision array, dimension (min(M,N)-1)
    The off-diagonal elements of the bidiagonal matrix B:
    if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
    if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.

    TAUQ    (output) COMPLEX_16 array dimension (min(M,N))
    The scalar factors of the elementary reflectors which
    represent the orthogonal matrix Q. See Further Details.

    TAUP    (output) COMPLEX_16 array, dimension (min(M,N))
    The scalar factors of the elementary reflectors which
    represent the orthogonal matrix P. See Further Details.

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
    On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    LWORK   (input) INTEGER
    The length of the array WORK. LWORK >= (M+N)*NB, where NB
    is the optimal blocksize.

    If LWORK = -1, then a workspace query is assumed; the routine
    only calculates the optimal size of the WORK array, returns
    this value as the first entry of the WORK array, and no error
    message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
    = 0:  successful exit
    < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============
    The matrices Q and P are represented as products of elementary
    reflectors:

    If m >= n,
    Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
    Each H(i) and G(i) has the form:
    H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'
    where tauq and taup are complex scalars, and v and u are complex vectors;
    v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);
    u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);
    tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n,
    Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)
    Each H(i) and G(i) has the form:
    H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'
    where tauq and taup are complex scalars, and v and u are complex vectors;
    v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in A(i+2:m,i);
    u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in A(i,i+1:n);
    tauq is stored in TAUQ(i) and taup in TAUP(i).

    The contents of A on exit are illustrated by the following examples:

    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

    ( d   e   u1  u1  u1)           ( d   u1  u1  u1  u1  u1)
    ( v1  d   e   u2  u2)           ( e   d   u2  u2  u2  u2)
    ( v1  v2  d   e   u3)           ( v1  e   d   u3  u3  u3)
    ( v1  v2  v3  d   e )           ( v1  v2  e   d   u4  u4)
    ( v1  v2  v3  v4  d )           ( v1  v2  v3  e   d   u5)
    ( v1  v2  v3  v4  v5)

    where d and e denote diagonal and off-diagonal elements of B, vi
    denotes an element of the vector defining H(i), and ui an element of
    the vector defining G(i).
    =====================================================================    */

    typedef typename af::dtype_traits<Ty>::base_type Tr;

    Tr *d = (Tr *)_d;
    Tr *e = (Tr *)_e;


    Ty c_neg_one = magma_neg_one<Ty>();
    Ty c_one     = magma_one<Ty>();
    cl_mem dwork;

    magma_int_t ncol, nrow, jmax, nb;

    magma_int_t i, j, nx;
    //magma_int_t iinfo;

    magma_int_t minmn;
    magma_int_t ldwrkx, ldwrky, lwkopt;
    magma_int_t lquery;

    nb   = magma_get_gebrd_nb<Ty>(n);

    lwkopt = (m + n) * nb;
    work[0] = magma_make<Ty>(lwkopt, 0.);
    lquery = (lwork == -1);

    /* Check arguments */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < std::max(1,m)) {
        *info = -4;
    } else if (lwork < lwkopt && (! lquery)) {
        *info = -10;
    }
    if (*info < 0) {
        //magma_xerbla(__func__, -(*info));
        return *info;
    }
    else if (lquery)
        return *info;

    /* Quick return if possible */
    minmn = std::min(m,n);
    if (minmn == 0) {
        work[0] = c_one;
        return *info;
    }

    if (MAGMA_SUCCESS != magma_malloc<Ty>(&dwork, (m + n)*nb)) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    size_t dwork_offset = 0;

    cl_event event =  0;

    ldwrkx = m;
    ldwrky = n;

    /* Set the block/unblock crossover point NX. */
    nx = 128;

    /* Copy the matrix to the GPU */
    if (copy && minmn - nx >= 1) {
        magma_setmatrix<Ty>(m, n, a, lda, da, da_offset, ldda, queue);
    }

    gpu_blas_gemm_func<Ty> gpu_blas_gemm;
    cpu_lapack_gebrd_work_func<Ty> cpu_lapack_gebrd_work;

    for (i=0; i< (minmn - nx); i += nb) {
        /*  Reduce rows and columns i:i+nb-1 to bidiagonal form and return
            the matrices X and Y which are needed to update the unreduced
            part of the matrix */
        nrow = m - i;
        ncol = n - i;

        /*   Get the current panel (no need for the 1st iteration) */
        if (i > 0) {
            magma_getmatrix<Ty>(nrow, nb, dA(i, i), ldda, A(i, i), lda, queue);
            magma_getmatrix<Ty>(nb, ncol - nb,
                                dA(i, i+nb), ldda,
                                A(i, i+nb), lda, queue);
        }

        magma_labrd_gpu<Ty>(nrow, ncol, nb,
                            A(i, i),          lda,
                            dA(i, i),          ldda,
                            d+i, e+i, tauq+i, taup+i,
                            work,             ldwrkx, dwork, dwork_offset,             ldwrkx,  // x, dx
                            work+(ldwrkx*nb), ldwrky, dwork, dwork_offset+(ldwrkx*nb), ldwrky,  // y, dy
                            queue);

        /*  Update the trailing submatrix A(i+nb:m,i+nb:n), using an update
            of the form  A := A - V*Y' - X*U' */
        nrow = m - i - nb;
        ncol = n - i - nb;

        // Send Y back to the GPU
        magma_setmatrix<Ty>(nrow, nb, work+nb, ldwrkx, dwork, dwork_offset+nb, ldwrkx, queue);
        magma_setmatrix<Ty>(ncol, nb,
                            work  +               (ldwrkx+1)*nb, ldwrky,
                            dwork, dwork_offset + (ldwrkx+1)*nb, ldwrky, queue);

        CLBLAS_CHECK(gpu_blas_gemm(clblasNoTrans, clblasConjTrans,
                                   nrow, ncol, nb,
                                   c_neg_one, dA(i+nb, i  ),      ldda,
                                   dwork, dwork_offset+(ldwrkx+1)*nb, ldwrky,
                                   c_one,     dA(i+nb, i+nb), ldda,
                                   1, &queue, 0, nullptr, &event));

        CLBLAS_CHECK(gpu_blas_gemm(clblasNoTrans, clblasNoTrans,
                                   nrow, ncol, nb,
                                   c_neg_one, dwork, dwork_offset+nb, ldwrkx,
                                   dA(i,    i+nb), ldda,
                                   c_one,     dA(i+nb, i+nb), ldda,
                                   1, &queue, 0, nullptr, &event));

        /* Copy diagonal and off-diagonal elements of B back into A */
        if (m >= n) {
            jmax = i + nb;
            for (j = i; j < jmax; ++j) {
                *A(j, j ) = magma_make<Ty>(d[j], 0.);
                *A(j, j+1) = magma_make<Ty>(e[j], 0.);
            }
        } else {
            jmax = i + nb;
            for (j = i; j < jmax; ++j) {
                *A(j,   j) = magma_make<Ty>(d[j], 0.);
                *A(j+1, j) = magma_make<Ty>(e[j], 0.);
            }
        }
    }

    /* Use unblocked code to reduce the remainder of the matrix */
    nrow = m - i;
    ncol = n - i;

    if (0 < minmn - nx) {
        magma_getmatrix<Ty>(nrow, ncol, dA(i, i), ldda, A(i, i), lda, queue);
    }

    LAPACKE_CHECK(cpu_lapack_gebrd_work(nrow, ncol,
                                        A(i, i), lda, d+i, e+i,
                                        tauq+i, taup+i, work, lwork));
    work[0] = magma_make<Ty>(lwkopt, 0.);

    magma_free(dwork);
    *info = 0;
    return 0;
} /* magma_zgebrd */

#define INSTANTIATE(Ty)                                 \
    template magma_int_t                                \
    magma_gebrd_hybrid<Ty>(                             \
        magma_int_t m, magma_int_t n,                   \
        Ty *a, magma_int_t lda,                         \
        cl_mem da, size_t da_offset, magma_int_t ldda,  \
        void *_d, void *_e,                             \
        Ty *tauq, Ty *taup,                             \
        Ty *work, magma_int_t lwork,                    \
        magma_queue_t queue,                            \
        magma_int_t *info, bool copy);                  \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
