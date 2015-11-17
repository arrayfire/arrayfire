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

template<typename Ty>  magma_int_t
magma_labrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    Ty *a, magma_int_t lda,
    cl_mem da, size_t da_offset, magma_int_t ldda,
    void *_d, void *_e, Ty *tauq, Ty *taup,
    Ty *x, magma_int_t ldx,
    cl_mem dx, size_t dx_offset, magma_int_t lddx,
    Ty *y, magma_int_t ldy,
    cl_mem dy, size_t dy_offset, magma_int_t lddy,
    magma_queue_t queue)
{
/*  -- MAGMA (version 1.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    @date

    Purpose
    =======
    ZLABRD reduces the first NB rows and columns of a complex general
    m by n matrix A to upper or lower bidiagonal form by an orthogonal
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by SGEBRD

    Arguments
    =========
    M       (input) INTEGER
    The number of rows in the matrix A.

    N       (input) INTEGER
    The number of columns in the matrix A.

    NB      (input) INTEGER
    The number of leading rows and columns of A to be reduced.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
    On entry, the m by n general matrix to be reduced.
    On exit, the first NB rows and columns of the matrix are
    overwritten; the rest of the array is unchanged.
    If m >= n, elements on and below the diagonal in the first NB
    columns, with the array TAUQ, represent the orthogonal
    matrix Q as a product of elementary reflectors; and
    elements above the diagonal in the first NB rows, with the
    array TAUP, represent the orthogonal matrix P as a product
    of elementary reflectors.
    If m < n, elements below the diagonal in the first NB
    columns, with the array TAUQ, represent the orthogonal
    matrix Q as a product of elementary reflectors, and
    elements on and above the diagonal in the first NB rows,
    with the array TAUP, represent the orthogonal matrix P as
    a product of elementary reflectors.
    See Further Details.

    LDA     (input) INTEGER
    The leading dimension of the array A.  LDA >= max(1,M).

    D       (output) COMPLEX_16 array, dimension (NB)
    The diagonal elements of the first NB rows and columns of
    the reduced matrix.  D(i) = A(i,i).

    E       (output) COMPLEX_16 array, dimension (NB)
    The off-diagonal elements of the first NB rows and columns of
    the reduced matrix.

    TAUQ    (output) COMPLEX_16 array dimension (NB)
    The scalar factors of the elementary reflectors which
    represent the orthogonal matrix Q. See Further Details.

    TAUP    (output) COMPLEX_16 array, dimension (NB)
    The scalar factors of the elementary reflectors which
    represent the orthogonal matrix P. See Further Details.

    X       (output) COMPLEX_16 array, dimension (LDX,NB)
    The m-by-nb matrix X required to update the unreduced part
    of A.

    LDX     (input) INTEGER
    The leading dimension of the array X. LDX >= M.

    Y       (output) COMPLEX_16 array, dimension (LDY,NB)
    The n-by-nb matrix Y required to update the unreduced part
    of A.

    LDY     (input) INTEGER
    The leading dimension of the array Y. LDY >= N.

    Further Details
    ===============
    The matrices Q and P are represented as products of elementary
    reflectors:

    Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb)

    Each H(i) and G(i) has the form:

    H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are complex scalars, and v and u are complex vectors.

    If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in
    A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in
    A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    The elements of the vectors v and u together form the m-by-nb matrix
    V and the nb-by-n matrix U' which are needed, with X and Y, to apply
    the transformation to the unreduced part of the matrix, using a block
    update of the form:  A := A - V*Y' - X*U'.

    The contents of A on exit are illustrated by the following examples
    with nb = 2:

    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

    ( 1   1   u1  u1  u1)           ( 1   u1  u1  u1  u1  u1)
    ( v1  1   1   u2  u2)           ( 1   1   u2  u2  u2  u2)
    ( v1  v2  a   a   a )           ( v1  1   a   a   a   a )
    ( v1  v2  a   a   a )           ( v1  v2  a   a   a   a )
    ( v1  v2  a   a   a )           ( v1  v2  a   a   a   a )
    ( v1  v2  a   a   a )

    where a denotes an element of the original matrix which is unchanged,
    vi denotes an element of the vector defining H(i), and ui an element
    of the vector defining G(i).
    =====================================================================    */

    typedef typename af::dtype_traits<Ty>::base_type Tr;

    const bool is_cplx = opencl::is_complex<Ty>::value;

    Tr *d = (Tr *)_d;
    Tr *e = (Tr *)_e;

    Ty c_neg_one = magma_neg_one<Ty>();
    Ty c_one = magma_one<Ty>();
    Ty c_zero = magma_zero<Ty>();
    magma_int_t c__1 = 1;

    magma_int_t a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__2, i__3;
    magma_int_t i__;
    Ty alpha;

    a_dim1 = lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d;
    --e;
    --tauq;
    --taup;

    x_dim1 = ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    dx_offset -= 1 + lddx;

    y_dim1 = ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;
    dy_offset -= 1 + lddy;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return 0;
    }

    Ty *f;
    magma_malloc_cpu<Ty>(&f, std::max(n,m));
    assert(f != NULL);  // TODO return error, or allocate outside zlatrd

    magma_event_t event = NULL;

    gpu_blas_gemv_func<Ty> gpu_blas_gemv;
    cpu_blas_gemv_func<Ty> cpu_blas_gemv;
    cpu_blas_scal_func<Ty> cpu_blas_scal;
    cpu_blas_axpy_func<Ty> cpu_blas_axpy;
    cpu_lapack_larfg_work_func<Ty> cpu_lapack_larfg;
    cpu_lapack_lacgv_work_func<Ty> cpu_lapack_lacgv;

    CBLAS_TRANSPOSE CblasTransParam = is_cplx ? CblasConjTrans : CblasTrans;

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i__ = 1; i__ <= nb; ++i__) {
            /*  Update A(i:m,i) */
            i__2 = m - i__ + 1;
            i__3 = i__ - 1;

            if (is_cplx) {
                LAPACKE_CHECK(cpu_lapack_lacgv(i__3, &y[i__+y_dim1], ldy));
            }

            cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one), &a[i__ + a_dim1], lda,
                          &y[i__+y_dim1], ldy, cblas_scalar(&c_one), &a[i__ + i__ * a_dim1], c__1);

            if (is_cplx) {
                LAPACKE_CHECK(cpu_lapack_lacgv(i__3, &y[i__+y_dim1], ldy));
            }

            cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one), &x[i__ + x_dim1], ldx,
                          &a[i__*a_dim1+1], c__1, cblas_scalar(&c_one), &a[i__+i__*a_dim1], c__1);

            /* Generate reflection Q(i) to annihilate A(i+1:m,i) */
            alpha = a[i__ + i__ * a_dim1];
            i__2 = m - i__ + 1;
            i__3 = i__ + 1;

            LAPACKE_CHECK(cpu_lapack_larfg(i__2, &alpha,
                                          &a[std::min(i__3,m) + i__ * a_dim1],
                                          c__1, &tauq[i__]));

            d[i__] = magma_real<Ty>(alpha);
            if (i__ < n) {
                a[i__ + i__ * a_dim1] = c_one;

                /* Compute Y(i+1:n,i) */
                i__2 = m - i__ + 1;
                i__3 = n - i__;

                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_setvector<Ty>(i__2,
                                    a + i__   + i__   * a_dim1, 1,
                                    da, da_offset + (i__-1)+(i__-1)* (ldda), 1,
                                    queue);
                // 2. Multiply ---------------------------------------------
                CLBLAS_CHECK(gpu_blas_gemv(clblasConjTrans, i__2, i__3, c_one,
                                           da, da_offset + (i__-1) + ((i__-1) + 1) * (ldda), ldda,
                                           da, da_offset + (i__-1) + (i__-1) * (ldda), c__1, c_zero,
                                           dy, dy_offset + i__ + 1 + i__ * y_dim1, c__1,
                                           1, &queue, 0, nullptr, &event));

                // 3. Put the result back ----------------------------------
                magma_getmatrix_async<Ty>(i__3, 1,
                                          dy, dy_offset + i__+1+i__*y_dim1, y_dim1,
                                          y+i__+1+i__*y_dim1, y_dim1,
                                          queue, &event);
                i__2 = m - i__ + 1;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasTransParam, i__2, i__3, cblas_scalar(&c_one), &a[i__ + a_dim1],
                              lda, &a[i__ + i__ * a_dim1], c__1, cblas_scalar(&c_zero),
                              &y[i__ * y_dim1 + 1], c__1);

                i__2 = n - i__;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one), &y[i__ + 1 +y_dim1], ldy,
                              &y[i__ * y_dim1 + 1], c__1,
                              cblas_scalar(&c_zero), f, c__1);
                i__2 = m - i__ + 1;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasTransParam, i__2, i__3, cblas_scalar(&c_one), &x[i__ + x_dim1],
                              ldx, &a[i__ + i__ * a_dim1], c__1, cblas_scalar(&c_zero),
                              &y[i__ * y_dim1 + 1], c__1);

                // 4. Synch to make sure the result is back ----------------
                magma_event_sync(event);

                if (i__3 != 0){
                    i__2 = n - i__;
                    cpu_blas_axpy(i__2, cblas_scalar(&c_one), f,c__1, &y[i__+1+i__*y_dim1],c__1);
                }

                i__2 = i__ - 1;
                i__3 = n - i__;
                cpu_blas_gemv(CblasTransParam, i__2, i__3, cblas_scalar(&c_neg_one),
                              &a[(i__ + 1) * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], c__1, cblas_scalar(&c_one),
                              &y[i__ + 1 + i__ * y_dim1], c__1);
                i__2 = n - i__;
                cpu_blas_scal(i__2, cblas_scalar(&tauq[i__]), &y[i__ + 1 + i__ * y_dim1], c__1);

                /* Update A(i,i+1:n) */
                i__2 = n - i__;
                if (is_cplx) {
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__2, &a[i__+(i__+1)*a_dim1], lda));
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__,  &a[i__+a_dim1], lda));
                }

                cpu_blas_gemv(CblasNoTrans, i__2, i__, cblas_scalar(&c_neg_one),
                              &y[i__ + 1 + y_dim1], ldy, &a[i__ + a_dim1], lda,
                              cblas_scalar(&c_one), &a[i__ + (i__ + 1) * a_dim1], lda);
                i__2 = i__ - 1;
                i__3 = n - i__;

                if (is_cplx) {
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__,  &a[i__+a_dim1], lda));
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__2, &x[i__+x_dim1], ldx));
                }

                cpu_blas_gemv(CblasTransParam, i__2, i__3, cblas_scalar(&c_neg_one), &a[(i__ + 1) *
                                                                              a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, cblas_scalar(&c_one), &a[
                                                                                  i__ + (i__ + 1) * a_dim1], lda);
                if (is_cplx) {
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__2, &x[i__+x_dim1], ldx));
                }

                /* Generate reflection P(i) to annihilate A(i,i+2:n) */
                i__2 = n - i__;
                /* Computing MIN */
                i__3 = i__ + 2;
                alpha = a[i__ + (i__ + 1) * a_dim1];
                LAPACKE_CHECK(cpu_lapack_larfg(i__2, &alpha,
                                              &a[i__ + std::min(i__3,n) * a_dim1],
                                              lda, &taup[i__]));
                e[i__] = magma_real<Ty>(alpha);
                a[i__ + (i__ + 1) * a_dim1] = c_one;

                /* Compute X(i+1:m,i) */
                i__2 = m - i__;
                i__3 = n - i__;
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_setvector<Ty>(i__3,
                                    a + i__   + (i__   +1)* a_dim1, lda,
                                    da, da_offset + (i__-1)+((i__-1)+1)*(ldda), ldda,
                                    queue);
                // 2. Multiply ---------------------------------------------
                //magma_zcopy(i__3, da+(i__-1)+((i__-1)+1)*(ldda), ldda,
                //            dy + 1 + lddy, 1);
                CLBLAS_CHECK(gpu_blas_gemv(clblasNoTrans, i__2, i__3, c_one,
                                           da, da_offset + (i__-1)+1+ ((i__-1)+1) * (ldda), ldda,
                                           da, da_offset + (i__-1) +  ((i__-1)+1) * (ldda), ldda,
                                           //dy + 1 + lddy, 1,
                                           c_zero, dx, dx_offset + i__ + 1 + i__ * x_dim1, c__1,
                                           1, &queue, 0, nullptr, &event));

                // 3. Put the result back ----------------------------------
                magma_getmatrix_async<Ty>(i__2, 1,
                                          dx, dx_offset + i__+1+i__*x_dim1, x_dim1,
                                          x+i__+1+i__*x_dim1, x_dim1,
                                          queue, &event);

                i__2 = n - i__;
                cpu_blas_gemv(CblasTransParam, i__2, i__, cblas_scalar(&c_one), &y[i__ + 1 + y_dim1],
                              ldy, &a[i__ + (i__ + 1) * a_dim1], lda, cblas_scalar(&c_zero), &x[
                                  i__ * x_dim1 + 1], c__1);

                i__2 = m - i__;
                cpu_blas_gemv(CblasNoTrans, i__2, i__, cblas_scalar(&c_neg_one), &a[i__ + 1 + a_dim1], lda,
                              &x[i__ * x_dim1 + 1], c__1, cblas_scalar(&c_zero), f, c__1);
                i__2 = i__ - 1;
                i__3 = n - i__;
                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_one), &a[(i__ + 1) * a_dim1 + 1],
                              lda, &a[i__ + (i__ + 1) * a_dim1], lda,
                              cblas_scalar(&c_zero), &x[i__ * x_dim1 + 1], c__1);

                // 4. Synch to make sure the result is back ----------------
                magma_event_sync(event);

                if (i__!=0){
                    i__2 = m - i__;
                    cpu_blas_axpy(i__2, cblas_scalar(&c_one), f,c__1, &x[i__+1+i__*x_dim1],c__1);
                }


                i__2 = m - i__;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one), &x[i__ + 1 +
                                                                           x_dim1], ldx, &x[i__ * x_dim1 + 1], c__1, cblas_scalar(&c_one), &x[
                                                                               i__ + 1 + i__ * x_dim1], c__1);
                i__2 = m - i__;
                cpu_blas_scal(i__2, cblas_scalar(&taup[i__]), &x[i__ + 1 + i__ * x_dim1], c__1);

                if (is_cplx) {
                    i__2 = n - i__;
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__2,  &a[i__+(i__+1)*a_dim1], lda));
                    // 4. Send the block reflector  A(i+1:m,i) to the GPU after ZLACGV()
                    magma_setvector<Ty>(i__2,
                                        a + i__   + (i__   +1)* a_dim1, lda,
                                        da, da_offset + (i__-1)+((i__-1)+1)*(ldda), ldda,
                                        queue);
                }
            }
        }
    }
    else {
        /* Reduce to lower bidiagonal form */
        for (i__ = 1; i__ <= nb; ++i__) {

            /* Update A(i,i:n) */
            i__2 = n - i__ + 1;
            i__3 = i__ - 1;
            if (is_cplx) {
                LAPACKE_CHECK(cpu_lapack_lacgv(i__2, &a[i__ + i__ * a_dim1], lda));
                LAPACKE_CHECK(cpu_lapack_lacgv(i__3, &a[i__ + a_dim1], lda));
            }
            cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one), &y[i__ + y_dim1], ldy,
                          &a[i__ + a_dim1], lda, cblas_scalar(&c_one), &a[i__ + i__ * a_dim1], lda);
            i__2 = i__ - 1;
            if (is_cplx) {
                LAPACKE_CHECK(cpu_lapack_lacgv(i__3, &a[i__ + a_dim1], lda));
                LAPACKE_CHECK(cpu_lapack_lacgv(i__3, &x[i__ + x_dim1], ldx));
            }
            i__3 = n - i__ + 1;
            cpu_blas_gemv(CblasTransParam, i__2, i__3, cblas_scalar(&c_neg_one), &a[i__ * a_dim1 + 1],
                          lda, &x[i__ + x_dim1], ldx, cblas_scalar(&c_one), &a[i__ + i__ * a_dim1], lda);
            if (is_cplx) {
                LAPACKE_CHECK(cpu_lapack_lacgv(i__2, &x[i__ + x_dim1], ldx));
            }

            /* Generate reflection P(i) to annihilate A(i,i+1:n) */
            i__2 = n - i__ + 1;
            /* Computing MIN */
            i__3 = i__ + 1;
            alpha = a[i__ + i__ * a_dim1];
            LAPACKE_CHECK(cpu_lapack_larfg(i__2, &alpha,
                                           &a[i__ + std::min(i__3,n) * a_dim1], lda, &taup[i__]));
            d[i__] = magma_real<Ty>(alpha);
            if (i__ < m) {
                a[i__ + i__ * a_dim1] = c_one;

                /* Compute X(i+1:m,i) */
                i__2 = m - i__;
                i__3 = n - i__ + 1;

                // 1. Send the block reflector  A(i,i+1:n) to the GPU ------
                magma_setvector<Ty>(i__3,
                                    a + i__   + i__   * a_dim1, lda,
                                    da, da_offset + (i__-1)+(i__-1)* (ldda), ldda,
                                    queue);

                // 2. Multiply ---------------------------------------------
                //magma_zcopy(i__3, da+(i__-1)+(i__-1)*(ldda), ldda,
                //            dy + 1 + lddy, 1);
                CLBLAS_CHECK(gpu_blas_gemv(clblasNoTrans, i__2, i__3, c_one,
                                           da, da_offset + (i__-1)+1 + (i__-1) * ldda, ldda,
                                           da, da_offset + (i__-1)   + (i__-1) * ldda, ldda,
                                           // dy + 1 + lddy, 1,
                                           c_zero,
                                           dx, dx_offset + i__ + 1 + i__ * x_dim1, c__1,
                                           1, &queue, 0, nullptr, &event));


                // 3. Put the result back ----------------------------------
                magma_getmatrix_async<Ty>(i__2, 1,
                                          dx, dx_offset + i__+1+i__*x_dim1, x_dim1,
                                          x+i__+1+i__*x_dim1, x_dim1,
                                          queue, &event);

                i__2 = n - i__ + 1;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasTransParam, i__2, i__3, cblas_scalar(&c_one), &y[i__ + y_dim1],
                              ldy, &a[i__ + i__ * a_dim1], lda, cblas_scalar(&c_zero),
                              &x[i__ *  x_dim1 + 1], c__1);
                i__2 = m - i__;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one),
                              &a[i__ + 1 + a_dim1], lda, &x[i__ * x_dim1 + 1], c__1, cblas_scalar(&c_zero),
                              f, c__1);

                i__2 = i__ - 1;
                i__3 = n - i__ + 1;
                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_one),
                              &a[i__ * a_dim1 + 1], lda, &a[i__ + i__ * a_dim1], lda, cblas_scalar(&c_zero),
                              &x[i__ * x_dim1 + 1], c__1);

                // 4. Synch to make sure the result is back ----------------
                magma_event_sync(event);
                if (i__2 != 0){
                    i__3 = m - i__;
                    cpu_blas_axpy(i__3, cblas_scalar(&c_one), f,c__1, &x[i__+1+i__*x_dim1],c__1);
                }

                i__2 = m - i__;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one),
                              &x[i__ + 1 + x_dim1], ldx, &x[i__ * x_dim1 + 1], c__1, cblas_scalar(&c_one),
                              &x[i__ + 1 + i__ * x_dim1], c__1);
                i__2 = m - i__;
                cpu_blas_scal(i__2, cblas_scalar(&taup[i__]), &x[i__ + 1 + i__ * x_dim1], c__1);
                i__2 = n - i__ + 1;

                if (is_cplx) {
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__2, &a[i__ + i__ * a_dim1], lda));
                    magma_setvector<Ty>(i__2,
                                        a + i__   + (i__ )* a_dim1, lda,
                                        da, da_offset + (i__-1)+ (i__-1)*(ldda), ldda,
                                        queue);
                }

                /* Update A(i+1:m,i) */
                i__2 = m - i__;
                i__3 = i__ - 1;

                if (is_cplx) {
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__3, &y[i__ + y_dim1], ldy));
                }

                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one),
                              &a[i__ + 1 + a_dim1], lda, &y[i__ + y_dim1], ldy, cblas_scalar(&c_one),
                              &a[i__ + 1 + i__ * a_dim1], c__1);
                i__2 = m - i__;
                if (is_cplx) {
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__3, &y[i__ + y_dim1], ldy));
                }
                cpu_blas_gemv(CblasNoTrans, i__2, i__, cblas_scalar(&c_neg_one),
                              &x[i__ + 1 + x_dim1], ldx, &a[i__ * a_dim1 + 1], c__1, cblas_scalar(&c_one),
                              &a[i__ + 1 + i__ * a_dim1], c__1);

                /* Generate reflection Q(i) to annihilate A(i+2:m,i) */
                i__2 = m - i__;
                i__3 = i__ + 2;
                alpha = a[i__ + 1 + i__ * a_dim1];
                LAPACKE_CHECK(cpu_lapack_larfg(i__2, &alpha,
                                               &a[std::min(i__3,m) + i__ * a_dim1],
                                               c__1, &tauq[i__]));
                e[i__] = magma_real<Ty>(alpha);
                a[i__ + 1 + i__ * a_dim1] = c_one;

                /* Compute Y(i+1:n,i) */
                i__2 = m - i__;
                i__3 = n - i__;

                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_setvector<Ty>(i__2,
                                    a + i__   +1+  i__   * a_dim1, 1,
                                    da, da_offset + (i__-1)+1+ (i__-1)*(ldda),  1,
                                    queue);
                // 2. Multiply ---------------------------------------------
                CLBLAS_CHECK(gpu_blas_gemv(clblasConjTrans, i__2, i__3, c_one,
                                           da, da_offset + (i__-1)+1+ ((i__-1)+1) * ldda, ldda,
                                           da, da_offset + (i__-1)+1+  (i__-1)    * ldda, c__1,
                                           c_zero, dy, dy_offset + i__ + 1 + i__ * y_dim1, c__1,
                                           1, &queue, 0, nullptr, &event));

                // 3. Put the result back ----------------------------------
                magma_getmatrix_async<Ty>(i__3, 1,
                                          dy, dy_offset + i__+1+i__*y_dim1, y_dim1,
                                          y+i__+1+i__*y_dim1, y_dim1,
                                          queue, &event);

                i__2 = m - i__;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasTransParam, i__2, i__3, cblas_scalar(&c_one), &a[i__ + 1 + a_dim1],
                              lda, &a[i__ + 1 + i__ * a_dim1], c__1, cblas_scalar(&c_zero),
                              &y[ i__ * y_dim1 + 1], c__1);
                i__2 = n - i__;
                i__3 = i__ - 1;
                cpu_blas_gemv(CblasNoTrans, i__2, i__3, cblas_scalar(&c_neg_one),
                              &y[i__ + 1 + y_dim1], ldy, &y[i__ * y_dim1 + 1], c__1,
                              cblas_scalar(&c_zero), f, c__1);

                i__2 = m - i__;
                cpu_blas_gemv(CblasTransParam, i__2, i__, cblas_scalar(&c_one), &x[i__ + 1 + x_dim1],
                              ldx, &a[i__ + 1 + i__ * a_dim1], c__1, cblas_scalar(&c_zero),
                              &y[i__ * y_dim1 + 1], c__1);

                // 4. Synch to make sure the result is back ----------------
                magma_event_sync(event);
                if (i__3 != 0){
                    i__2 = n - i__;
                    cpu_blas_axpy(i__2, cblas_scalar(&c_one), f,c__1, &y[i__+1+i__*y_dim1],c__1);
                }

                i__2 = n - i__;
                cpu_blas_gemv(CblasTransParam, i__, i__2, cblas_scalar(&c_neg_one),
                              &a[(i__ + 1) * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1],
                              c__1, cblas_scalar(&c_one), &y[i__ + 1 + i__ * y_dim1], c__1);
                i__2 = n - i__;
                cpu_blas_scal(i__2, cblas_scalar(&tauq[i__]), &y[i__ + 1 + i__ * y_dim1], c__1);
            }
            else {
                if (is_cplx) {
                    i__2 = n - i__ + 1;
                    LAPACKE_CHECK(cpu_lapack_lacgv(i__2, &a[i__ + i__ * a_dim1], lda));
                    magma_setvector<Ty>(i__2,
                                        a + i__   + (i__ )* a_dim1, lda,
                                        da, da_offset + (i__-1)+ (i__-1)*(ldda), ldda,
                                        queue);
                }
            }
        }
    }

    magma_queue_sync(queue);
    magma_free_cpu(f);

    return MAGMA_SUCCESS;
}

#define INSTANTIATE(Ty)                                 \
    template  magma_int_t                               \
    magma_labrd_gpu<Ty>(                                \
        magma_int_t m, magma_int_t n, magma_int_t nb,   \
        Ty *a, magma_int_t lda,                         \
        cl_mem da, size_t da_offset, magma_int_t ldda,  \
        void *_d, void *_e, Ty *tauq, Ty *taup,         \
        Ty *x, magma_int_t ldx,                         \
        cl_mem dx, size_t dx_offset, magma_int_t lddx,  \
        Ty *y, magma_int_t ldy,                         \
        cl_mem dy, size_t dy_offset, magma_int_t lddy,  \
        magma_queue_t queue);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
