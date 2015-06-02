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

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: 'a' is pointer to the current panel holding the
      Householder vectors for the QR factorization of the panel. This routine
      puts ones on the diagonal and zeros in the upper triangular part of 'a'.
      The upper triangular values are stored in work.

      Then, the inverse is calculated in place in work, so as a final result,
      work holds the inverse of the upper triangular diagonal block.
 */

template<typename Ty>
void split_diag_block(magma_int_t ib, Ty *a, magma_int_t lda, Ty *work)
{
    magma_int_t i, j;
    Ty *cola, *colw;
    static const Ty c_zero = magma_zero<Ty>();
    static const Ty c_one  = magma_one<Ty>();

    for(i=0; i<ib; i++){
        cola = a    + i*lda;
        colw = work + i*ib;
        for(j=0; j<i; j++){
            colw[j] = cola[j];
            cola[j] = c_zero;
        }
        colw[i] = cola[i];
        cola[i] = c_one;
    }
}

template<typename Ty> magma_int_t
magma_geqrf3_gpu(
    magma_int_t m, magma_int_t n,
    cl_mem dA, size_t dA_offset,  magma_int_t ldda,
    Ty *tau, cl_mem dT, size_t dT_offset,
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
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    This version stores the triangular dT matrices used in
    the block QR factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application
    of Q is much faster. Also, the upper triangular matrices for V have 0s
    in them. The corresponding parts of the upper triangular R are inverted
    and stored separately in dT.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    LDDA     (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    dT      (workspace/output)  COMPLEX_16 array on the GPU,
            dimension (2*MIN(M, N) + (N+31)/32*32 )*NB,
            where NB can be obtained through magma_get_zgeqrf_nb(M).
            It starts with MIN(M,N)*NB block that store the triangular T
            matrices, followed by the MIN(M,N)*NB block of the diagonal
            inverses for the R matrix. The rest of the array is used as workspace.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ===============
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).
    =====================================================================    */

    #define a_ref(a_1,a_2) dA, (dA_offset + (a_1) + (a_2)*(ldda))
    #define t_ref(a_1)     dT, (dT_offset + (a_1)*nb)
    #define d_ref(a_1)     dT, (dT_offset + (minmn + (a_1))*nb)
    #define dd_ref(a_1)    dT, (dT_offset + (2*minmn+(a_1))*nb)
    #define work_ref(a_1)  ( work + (a_1))
    #define hwork          ( work + (nb)*(m))

    magma_int_t i, k, minmn, old_i, old_ib, rows, cols;
    magma_int_t ib, nb;
    magma_int_t ldwork, lddwork, lwork, lhwork;
    Ty *work, *ut;

    /* check arguments */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < std::max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        //magma_xerbla( __func__, -(*info) );
        return *info;
    }

    k = minmn = std::min(m,n);
    if (k == 0)
        return *info;

    nb = magma_get_geqrf_nb<Ty>(m);

    lwork  = (m + n + nb)*nb;
    lhwork = lwork - m*nb;

    if (MAGMA_SUCCESS != magma_malloc_cpu<Ty>( &work, lwork )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    ut = hwork+nb*(n);
    memset(ut, 0, nb*nb*sizeof(Ty));

    magma_event_t event[2] = {NULL, NULL};

    ldwork = m;
    lddwork= n;

    geqrf_work_func<Ty> cpu_geqrf;
    larft_func<Ty> cpu_larft;

    if ( (nb > 1) && (nb < k) ) {
        /* Use blocked code initially */
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nb; i += nb) {
            ib = std::min(k-i, nb);
            rows = m -i;
            magma_getmatrix_async<Ty>(rows, ib,
                                     a_ref(i,i),  ldda,
                                     work_ref(i), ldwork, queue, &event[1]);
            if (i>0){
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                cols = n-old_i-2*old_ib;
                magma_larfb_gpu<Ty>(MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                   m-old_i, cols, old_ib,
                                   a_ref(old_i, old_i         ), ldda, t_ref(old_i), nb,
                                   a_ref(old_i, old_i+2*old_ib), ldda, dd_ref(0),    lddwork, queue);

                /* store the diagonal */
                magma_setmatrix_async<Ty>(old_ib, old_ib,
                                         ut, old_ib,
                                         d_ref(old_i), old_ib, queue, &event[0]);
            }

            magma_event_sync(event[1]);
            *info = cpu_geqrf(LAPACK_COL_MAJOR, rows, ib, work_ref(i), ldwork, tau+i, hwork, lhwork);

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            cpu_larft(LAPACK_COL_MAJOR,
                      *MagmaForwardStr, *MagmaColumnwiseStr,
                      rows, ib,
                      work_ref(i), ldwork,
                      tau+i, hwork, ib);

            /* Put 0s in the upper triangular part of a panel (and 1s on the
               diagonal); copy the upper triangular in ut and invert it. */
            if (i > 0) magma_event_sync(event[0]);
            //Change me
            split_diag_block<Ty>(ib, work_ref(i), ldwork, ut);
            magma_setmatrix<Ty>(rows, ib, work_ref(i), ldwork, a_ref(i,i), ldda, queue);

            if (i + ib < n) {
                /* Send the triangular factor T to the GPU */
                magma_setmatrix<Ty>(ib, ib, hwork, ib, t_ref(i), nb, queue);

                if (i+nb < k-nb){
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    magma_larfb_gpu<Ty>(MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                       rows, ib, ib,
                                       a_ref(i, i   ), ldda, t_ref(i),  nb,
                                       a_ref(i, i+ib), ldda, dd_ref(0), lddwork, queue);
                }
                else {
                    cols = n-i-ib;
                    magma_larfb_gpu<Ty>(MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                       rows, cols, ib,
                                       a_ref(i, i   ), ldda, t_ref(i),  nb,
                                       a_ref(i, i+ib), ldda, dd_ref(0), lddwork, queue);
                    /* Fix the diagonal block */
                    magma_setmatrix<Ty>( ib, ib, ut, ib, d_ref(i), ib , queue);
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
        magma_getmatrix<Ty>( rows, ib, a_ref(i, i), ldda, work, rows, queue );

        lhwork = lwork - rows*ib;
        *info = cpu_geqrf(LAPACK_COL_MAJOR, rows, ib, work, rows, tau+i, work+ib*rows, lhwork);

        magma_setmatrix<Ty>( rows, ib, work, rows, a_ref(i, i), ldda, queue );
    }

    magma_free_cpu( work );
    return *info;
} /* magma_zgeqrf_gpu */

#undef a_ref
#undef t_ref
#undef d_ref
#undef work_ref

#define INSTANTIATE(T)                                  \
    template magma_int_t                                \
    magma_geqrf3_gpu<T>(                                 \
        magma_int_t m, magma_int_t n,                   \
        cl_mem dA, size_t dA_offset,  magma_int_t ldda, \
        T *tau, cl_mem dT, size_t dT_offset,            \
        magma_queue_t queue,                            \
        magma_int_t *info);                             \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
