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

#if 0  // Needs hetrd to be enabled
#include "magma.h"
#include "magma_blas.h"
#include "magma_data.h"
#include "magma_cpu_lapack.h"
#include "magma_helper.h"
#include "magma_sync.h"

#include <algorithm>

/**
    Purpose
    -------
    ZUNMQR overwrites the general complex M-by-N matrix C with

    @verbatim
                               SIDE = MagmaLeft    SIDE = MagmaRight
    TRANS = MagmaNoTrans:      Q * C               C * Q
    TRANS = Magma_ConjTrans:   Q**H * C            C * Q**H
    @endverbatim

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = MagmaLeft and of order N
    if SIDE = MagmaRight.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply Q or Q**H from the Left;
      -     = MagmaRight:     apply Q or Q**H from the Right.

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    No transpose, apply Q;
      -     = Magma_ConjTrans: Conjugate transpose, apply Q**H.

    @param[in]
    m       INTEGER
            The number of rows of the matrix C. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C. N >= 0.

    @param[in]
    k       INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = MagmaLeft,  M >= K >= 0;
            if SIDE = MagmaRight, N >= K >= 0.

    @param[in]
    dA      COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.
            The diagonal and the upper part
            are destroyed, the reflectors are not modified.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array DA.
            LDDA >= max(1,M) if SIDE = MagmaLeft; LDDA >= max(1,N) if SIDE = MagmaRight.

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    @param[in,out]
    dC      COMPLEX_16 array, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by (Q*C) or (Q**H * C) or (C * Q**H) or (C*Q).

    @param[in]
    lddc    INTEGER
            The leading dimension of the array C. LDDC >= max(1,M).

    @param[in]
    wA      (workspace) COMPLEX_16 array, dimension
                                 (LDWA,M) if SIDE = MagmaLeft
                                 (LDWA,N) if SIDE = MagmaRight
            The vectors which define the elementary reflectors, as
            returned by ZHETRD_GPU.

    @param[in]
    ldwa    INTEGER
            The leading dimension of the array wA.
            LDWA >= max(1,M) if SIDE = MagmaLeft; LDWA >= max(1,N) if SIDE = MagmaRight.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zgeqrf_comp
    ********************************************************************/
template<typename Ty> magma_int_t
magma_unmqr2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty    *tau,
    cl_mem dC, size_t dC_offset, magma_int_t lddc,
    Ty    *wA, magma_int_t ldwa,
    magma_queue_t queue,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA) , ((i_) + (j_)*ldda) + dA_offset
    #define dC(i_,j_) (dC) , ((i_) + (j_)*lddc) + dC_offset
    #define wA(i_,j_) (wA + (i_) + (j_)*ldwa)

    /* Allocate work space on the GPU */
    cl_mem dwork;

    static const Ty c_zero = magma_zero<Ty>();
    static const Ty c_one  = magma_one<Ty>();

    magma_int_t i, i__4, lddwork;
    Ty T[2*4160]        /* was [65][64] */;
    magma_int_t i1, i2, step, ib, ic, jc, nb, mi, ni, nq;
    int left, notran;

    wA -= 1 + ldwa;
    dC_offset -= 1 + lddc;
    --tau;

    *info = 0;
    left   = (side == MagmaLeft);
    notran = (trans == MagmaNoTrans);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        //nw = n;
        magma_malloc<Ty>( &dwork, (n + 64)*64 );  // TODO after checking args, else memory leak!
    } else {
        nq = n;
        //nw = m;
        magma_malloc<Ty>( &dwork, (m + 64)*64 );  // TODO after checking args, else memory leak!
    }
    if (! left && side != MagmaRight) {
        *info = -1;
    } else if (! notran && trans != Magma_ConjTrans) {
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
    } else if (ldwa < std::max(1,nq)) {
        *info = -12;
    }

    // size of the block
    nb = 64;

    if (*info != 0) {
        //magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return *info;
    }

    /* Use hybrid CPU-GPU code */
    if ( ( left && (! notran) ) ||  ( (! left) && notran ) ) {
        i1 = 1;
        i2 = k;
        step = nb;
    } else {
        i1 = ((k - 1)/nb)*nb + 1;
        i2 = 1;
        step = -nb;
    }

    // silence "uninitialized" warnings
    mi = 0;
    ni = 0;

    if (left) {
        ni = n;
        jc = 1;
    } else {
        mi = m;
        ic = 1;
    }

    larft_func<Ty> cpu_larft;

    // set nb-1 super-diagonals to 0, and diagonal to 1.
    // This way we can copy V directly to the GPU,
    // with the upper triangle parts already set to identity.
    magmablas_laset_band<Ty>( MagmaUpper, k, k, nb, c_zero, c_one, dA, dA_offset, ldda, queue);

    // for i=i1 to i2 by step
    for (i = i1; (step < 0 ? i >= i2 : i <= i2); i += step) {
        ib = std::min(nb, k - i + 1);

        /* Form the triangular factor of the block reflector
           H = H(i) H(i+1) . . . H(i+ib-1) */
        i__4 = nq - i + 1;
        cpu_larft(LAPACK_COL_MAJOR,
                  *MagmaForwardStr, *MagmaColumnwiseStr,
                  i__4, ib,
                  wA(i,i), ldwa, &tau[i], T, ib);

        if (left) {
            /* H or H' is applied to C(i:m,1:n) */
            mi = m - i + 1;
            ic = i;
        }
        else {
            /* H or H' is applied to C(1:m,i:n) */
            ni = n - i + 1;
            jc = i;
        }

        if (left)
            lddwork = ni;
        else
            lddwork = mi;

        /* Apply H or H'; First copy T to the GPU */
        magma_setmatrix<Ty>( ib, ib, T, ib, dwork, 0, ib, queue);
        magma_larfb_gpu<Ty>( side, trans, MagmaForward, MagmaColumnwise,
                             mi, ni, ib,
                             dA(i-1,i-1), ldda,
                             dwork, 0, ib,  // dA using 0-based indices here
                             dC(ic,jc), lddc,
                             dwork, ib*ib, lddwork, queue);
    }

    magma_free( dwork );

    return *info;
} /* magma_zunmqr */


#define INSTANTIATE(Ty)                                 \
    template magma_int_t                                \
    magma_unmqr2_gpu<Ty>(                               \
        magma_side_t side, magma_trans_t trans,         \
        magma_int_t m, magma_int_t n, magma_int_t k,    \
        cl_mem dA, size_t dA_offset, magma_int_t ldda,  \
        Ty    *tau,                                     \
        cl_mem dC, size_t dC_offset, magma_int_t lddc,  \
        Ty    *wA, magma_int_t ldwa,                    \
        magma_queue_t queue,                            \
        magma_int_t *info);                             \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
#endif
