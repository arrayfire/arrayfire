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
#include "../platform.hpp"

#include <algorithm>

template<typename Ty>
void panel_to_q(magma_uplo_t uplo, magma_int_t ib, Ty *A, magma_int_t lda, Ty *work)
{
    magma_int_t i, j, k = 0;
    Ty *col;
    static const Ty c_zero = magma_zero<Ty>();
    static const Ty c_one  = magma_one<Ty>();

    if (uplo == MagmaUpper) {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = 0; j < i; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }

            work[k] = col[i];
            col [j] = c_one;
            ++k;
        }
    }
    else {
        for(i=0; i<ib; ++i) {
            col = A + i*lda;
            work[k] = col[i];
            col [i] = c_one;
            ++k;
            for(j=i+1; j<ib; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
        }
    }
}


// -------------------------
// Restores a panel, after call to panel_to_q.
template<typename Ty>
void q_to_panel(magma_uplo_t uplo, magma_int_t ib, Ty *A, magma_int_t lda, Ty *work)
{
    magma_int_t i, j, k = 0;
    Ty *col;

    if (uplo == MagmaUpper) {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = 0; j <= i; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
    else {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = i; j < ib; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
}

template<typename Ty> magma_int_t
magma_geqrf2_gpu(
    magma_int_t m, magma_int_t n,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty *tau,
    magma_queue_t* queue,
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

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

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

    #define dA(a_1,a_2)    dA, (dA_offset + (a_1) + (a_2)*(ldda))
    #define work(a_1)      ( work + (a_1))
    #define hwork          ( work + (nb)*(m))

    cl_mem dwork;
    Ty  *work;

    magma_int_t i, k, ldwork, lddwork, old_i, old_ib, rows;
    magma_int_t nbmin, nx, ib, nb;
    magma_int_t lhwork, lwork;

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

    k = std::min(m,n);
    if (k == 0)
        return *info;

    nb = magma_get_geqrf_nb<Ty>(m);

    lwork  = (m+n) * nb;
    lhwork = lwork - (m)*nb;

    if ( MAGMA_SUCCESS != magma_malloc<Ty>( &dwork, n*nb )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    /*
    if ( MAGMA_SUCCESS != magma_zmalloc_cpu( &work, lwork ) ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        magma_free( dwork );
        return *info;
    }
    */

    cl_mem buffer = clCreateBuffer(opencl::getContext()(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(Ty)*lwork, NULL, NULL);
    work = (Ty*)clEnqueueMapBuffer(queue[0], buffer, CL_TRUE,
                                   CL_MAP_READ | CL_MAP_WRITE,
                                   0, lwork*sizeof(Ty),
                                   0, NULL, NULL, NULL);

    geqrf_work_func<Ty> cpu_geqrf;
    larft_func<Ty> cpu_larft;

    nbmin = 2;
    nx    = nb;
    ldwork = m;
    lddwork= n;

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code initially */
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nx; i += nb) {
            ib = std::min(k-i, nb);
            rows = m -i;

            magma_queue_sync( queue[1] );
            magma_getmatrix_async<Ty>(rows, ib, dA(i, i), ldda, work(i), ldwork, queue[0], NULL);

            if (i > 0) {
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_larfb_gpu<Ty>( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                     m-old_i, n-old_i-2*old_ib, old_ib,
                                     dA(old_i, old_i         ), ldda, dwork,0,      lddwork,
                                     dA(old_i, old_i+2*old_ib), ldda, dwork,old_ib, lddwork, queue[1]);

                magma_setmatrix_async<Ty>( old_ib, old_ib, work(old_i), ldwork,
                                           dA(old_i, old_i), ldda, queue[1], NULL);
            }

            magma_queue_sync(queue[0]);
            *info = cpu_geqrf(LAPACK_COL_MAJOR, rows, ib, work(i), ldwork, tau+i, hwork, lhwork);

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            cpu_larft(LAPACK_COL_MAJOR,
                      *MagmaForwardStr, *MagmaColumnwiseStr,
                      rows, ib,
                      work(i), ldwork, tau+i, hwork, ib);

            panel_to_q<Ty>( MagmaUpper, ib, work(i), ldwork, hwork+ib*ib );

            /* download the i-th V matrix */
            magma_setmatrix_async<Ty>(rows, ib, work(i), ldwork, dA(i,i), ldda, queue[0], NULL);

            /* download the T matrix */
            magma_queue_sync( queue[1] );
            magma_setmatrix_async<Ty>( ib, ib, hwork, ib, dwork, 0, lddwork, queue[0], NULL);
            magma_queue_sync( queue[0] );

            if (i + ib < n) {
                if (i+nb < k-nx) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    magma_larfb_gpu<Ty>( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                         rows, ib, ib,
                                         dA(i, i   ), ldda, dwork,0,  lddwork,
                                         dA(i, i+ib), ldda, dwork,ib, lddwork, queue[1]);
                    q_to_panel<Ty>( MagmaUpper, ib, work(i), ldwork, hwork+ib*ib );
                }
                else {
                    magma_larfb_gpu<Ty>( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                         rows, n-i-ib, ib,
                                         dA(i, i   ), ldda, dwork,0,  lddwork,
                                         dA(i, i+ib), ldda, dwork,ib, lddwork, queue[1]);
                    q_to_panel<Ty>( MagmaUpper, ib, work(i), ldwork, hwork+ib*ib );
                    magma_setmatrix_async<Ty>(ib, ib, work(i), ldwork, dA(i,i), ldda, queue[1], NULL);
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    magma_free(dwork);

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
        magma_getmatrix_async<Ty>(rows, ib, dA(i, i), ldda, work, rows, queue[1], NULL);
        magma_queue_sync(queue[1]);

        lhwork = lwork - rows*ib;
        *info = cpu_geqrf(LAPACK_COL_MAJOR, rows, ib, work, rows, tau+i, work+ib*rows, lhwork);

        magma_setmatrix_async<Ty>(rows, ib, work, rows, dA(i, i), ldda, queue[1], NULL);
    }

    magma_queue_sync(queue[0]);
    magma_queue_sync(queue[1]);

    // magma_free_cpu(work);
    clEnqueueUnmapMemObject(queue[0], buffer, work, 0, NULL, NULL);
    clReleaseMemObject(buffer);

    return *info;
} /* magma_zgeqrf2_gpu */

#define INSTANTIATE(Ty)                                 \
    template magma_int_t                                \
    magma_geqrf2_gpu<Ty>(                               \
        magma_int_t m, magma_int_t n,                   \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,  \
        Ty *tau,                                        \
        magma_queue_t* queue,                           \
        magma_int_t *info);                             \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
