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
#include "magma_data.h"
#include "magma_cpu_lapack.h"
#include "magma_helper.h"
#include "magma_sync.h"

#include <algorithm>

template<typename Ty>  magma_int_t
magma_unmqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty *tau,
    cl_mem dC, size_t dC_offset, magma_int_t lddc,
    Ty *hwork, magma_int_t lwork,
    cl_mem dT, size_t dT_offset, magma_int_t nb,
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
    ZUNMQR_GPU overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**H * C       C * Q**H

    where Q is a complex orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========
    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    DA      (input) COMPLEX_16 array on the GPU, dimension (LDDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument DA.
            DA is modified by the routine but restored on exit.

    LDDA    (input) INTEGER
            The leading dimension of the array DA.
            If SIDE = 'L', LDDA >= max(1,M);
            if SIDE = 'R', LDDA >= max(1,N).

    TAU     (input) COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    DC      (input/output) COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H * C or C * Q**H or C*Q.

    LDDC     (input) INTEGER
            The leading dimension of the array DC. LDDC >= max(1,M).

    HWORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, HWORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array HWORK.
            LWORK >= (M-K+NB)*(N+2*NB) if SIDE = 'L',
            and LWORK >= (N-K+NB)*(M+2*NB) if SIDE = 'R', where NB is the
            optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array, and no error
            message related to LWORK is issued by XERBLA.

    DT      (input) COMPLEX_16 array on the GPU that is the output
            (the 9th argument) of magma_zgeqrf_gpu.

    NB      (input) INTEGER
            This is the blocking size that was used in pre-computing DT, e.g.,
            the blocking size used in magma_zgeqrf_gpu.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    #define a_ref(a_1,a_2) dA, (dA_offset+(a_1)+(a_2)*(ldda))
    #define c_ref(a_1,a_2) dC, (dC_offset+(a_1)+(a_2)*(lddc))
    #define t_ref(a_1)     dT, (dT_offset+(a_1)*nb)

    static const Ty c_one = magma_one<Ty>();

    cl_mem dwork;
    magma_int_t i;

    magma_int_t i1, i2, step, ib, ic, jc, ma, mi, ni, nq, nw, ret;
    int left, notran, lquery;
    magma_int_t lwkopt;

    *info = 0;
    left   = (side == MagmaLeft);
    notran = (trans == MagmaNoTrans);
    lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }
    if ( (!left) && (side != MagmaRight) ) {
        *info = -1;
    } else if ( (!notran) && (trans != MagmaConjTrans) ) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (ldda < std::max(1,nq)) {
        *info = -7;
    } else if (lddc < std::max(1,m)) {
        *info = -10;
    } else if (lwork < std::max(1,nw) && ! lquery) {
        *info = -12;
    }

    lwkopt = (m-k+nb)*(n+2*nb);
    hwork[0] = magma_scalar<Ty>(lwkopt);

    if (*info != 0) {
        //magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        hwork[0] = c_one;
        return *info;
    }

    magma_malloc<Ty>(&dwork, (((n+31)/32)*32)*nb);

    cpu_lapack_unmqr_work_func<Ty> cpu_lapack_unmqr;

    if ( (left && (! notran)) || ( (!left) && notran ) ) {
        i1 = 0;
        i2 = k-nb;
        step = nb;
    } else {
        i1 = (k - 1 - nb) / nb * nb;
        i2 = 0;
        step = -nb;
    }

    mi = 0;
    ni = 0;

    if (left) {
        ni = n;
        jc = 0;
    } else {
        mi = m;
        ic = 0;
    }

    static const bool is_real = magma_is_real<Ty>();

    /* Use unblocked code to multiply last or only block (cases Q*C or C*Q^T). */
    // workspace left:  A(mi*nb) + C(mi*ni) + work(ni*nb_la) = (m-k-nb)*nb + (m-k-nb)*n + n*nb
    // workspace right: A(ni*nb) + C(mi*ni) + work(mi*nb_la) = (n-k-nb)*nb + m*(n-k-nb) + m*nb
    if ( step < 0 ) {
        // i is beginning of last block
        i = i1 - step;
        if ( i >= k ) {
            i = i1;
        }
        ib = k - i;
        if (left) {
            // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
            mi = m - i;
            ma = mi;
            ic = i;
        }
        else {
            // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
            ni = n - i;
            ma = ni;
            jc = i;
        }

        Ty* hA = hwork;
        Ty* hC = hwork + ma*ib;
        Ty* hW = hwork + ma*ib + mi*ni;
        magma_int_t lhwork = lwork - (ma*ib + mi*ni);

        magma_getmatrix<Ty>(ma, ib, a_ref(i,  i ), ldda, hA, ma, queue);
        magma_getmatrix<Ty>(mi, ni, c_ref(ic, jc), lddc, hC, mi, queue);

        LAPACKE_CHECK(cpu_lapack_unmqr(
                          side == MagmaRight ? 'R' : 'L',
                          notran ? 'N' : (is_real ? 'T' : 'C'),
                          mi, ni, ib,
                          hA, ma, tau+i,
                          hC, mi,
                          hW, lhwork));

        // send the updated part of C back to the GPU
        magma_setmatrix<Ty>( mi, ni, hC, mi, c_ref(ic, jc), lddc, queue);
    }


    if (nb < k)
    {
        for (i=i1; step<0 ? i>i2 : i<i2; i+=step)
        {
            ib = std::min(nb, k - i);
            if (left){
                mi = m - i;
                ic = i;
            }
            else {
                ni = n - i;
                jc = i;
            }

            if (mi == 0 || ni == 0) break;

            ret = magma_larfb_gpu<Ty>(MagmaLeft,
                                      is_real ? MagmaTrans : MagmaConjTrans,
                                      MagmaForward, MagmaColumnwise,
                                      mi, ni, ib,
                                      a_ref(i,  i ), ldda, t_ref(i), nb,
                                      c_ref(ic, jc), lddc, dwork, 0, nw, queue);
            if ( ret != MAGMA_SUCCESS )
              return ret;
        }
    }
    else
    {
        i = i1;
    }

    /* Use unblocked code to multiply the last or only block (cases Q^T*C or C*Q). */
    if ( step > 0 ) {
        ib = k-i;
        if (left) {
            // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
            mi = m - i;
            ma = mi;
            ic = i;
        }
        else {
            // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
            ni = n - i;
            ma = ni;
            jc = i;
        }

        Ty* hA = hwork;
        Ty* hC = hwork + ma*ib;
        Ty* hW = hwork + ma*ib + mi*ni;
        magma_int_t lhwork = lwork - (ma*ib + mi*ni);

        magma_getmatrix<Ty>(ma, ib, a_ref(i,  i ), ldda, hA, ma, queue);
        magma_getmatrix<Ty>(mi, ni, c_ref(ic, jc), lddc, hC, mi, queue);

        LAPACKE_CHECK(cpu_lapack_unmqr(
                          side == MagmaRight ? 'R' : 'L',
                          notran ? 'N' : (is_real ? 'T' : 'C'),
                          mi, ni, ib,
                          hA, ma, tau+i,
                          hC, mi,
                          hW, lhwork));

        // send the updated part of C back to the GPU
        magma_setmatrix<Ty>(mi, ni, hC, mi, c_ref(ic, jc), lddc, queue);
    }

    magma_free(dwork);

    return *info;
    /* End of MAGMA_ZUNMQR_GPU */
}

#define INSTANTIATE(T)                                  \
    template  magma_int_t                               \
    magma_unmqr_gpu<T>(                                 \
        magma_side_t side, magma_trans_t trans,         \
        magma_int_t m, magma_int_t n, magma_int_t k,    \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,  \
        T *tau,                                         \
        cl_mem dC, size_t dC_offset, magma_int_t lddc,  \
        T *hwork, magma_int_t lwork,                    \
        cl_mem dT, size_t dT_offset, magma_int_t nb,    \
        magma_queue_t queue,                            \
        magma_int_t *info);                             \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
