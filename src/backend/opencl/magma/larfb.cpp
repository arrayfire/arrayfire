/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates
       @precisions normal z -> s d c
*/

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
    ZLARFB applies a complex block reflector H or its transpose H^H to a
    COMPLEX_16 m by n matrix C, from the left.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply H or H^H from the Left
      -     = MagmaRight:     apply H or H^H from the Right

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    apply H   (No transpose)
      -     = Magma_ConjTrans: apply H^H (Conjugate transpose)

    @param[in]
    direct  magma_direct_t
            Indicates how H is formed from a product of elementary
            reflectors
      -     = MagmaForward:  H = H(1) H(2) . . . H(k) (Forward)
      -     = MagmaBackward: H = H(k) . . . H(2) H(1) (Backward)

    @param[in]
    storev  magma_storev_t
            Indicates how the vectors which define the elementary
            reflectors are stored:
      -     = MagmaColumnwise: Columnwise
      -     = MagmaRowwise:    Rowwise

    @param[in]
    m       INTEGER
            The number of rows of the matrix C.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C.

    @param[in]
    k       INTEGER
            The order of the matrix T (= the number of elementary
            reflectors whose product defines the block reflector).

    @param[in]
    dV      COMPLEX_16 array on the GPU, dimension
                (LDDV,K) if STOREV = MagmaColumnwise
                (LDDV,M) if STOREV = MagmaRowwise and SIDE = MagmaLeft
                (LDDV,N) if STOREV = MagmaRowwise and SIDE = MagmaRight
            The matrix V. See further details.

    @param[in]
    lddv    INTEGER
            The leading dimension of the array V.
            If STOREV = MagmaColumnwise and SIDE = MagmaLeft, LDDV >= max(1,M);
            if STOREV = MagmaColumnwise and SIDE = MagmaRight, LDDV >= max(1,N);
            if STOREV = MagmaRowwise, LDDV >= K.

    @param[in]
    dT      COMPLEX_16 array on the GPU, dimension (LDDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    @param[in]
    lddt    INTEGER
            The leading dimension of the array T. LDDT >= K.

    @param[in,out]
    dC      COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C, or H^H*C, or C*H, or C*H^H.

    @param[in]
    lddc    INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    @param
    dwork   (workspace) COMPLEX_16 array, dimension (LDWORK,K)

    @param[in]
    ldwork  INTEGER
            The leading dimension of the array WORK.
            If SIDE = MagmaLeft,  LDWORK >= max(1,N);
            if SIDE = MagmaRight, LDWORK >= max(1,M);

    Further Details
    ---------------
    The shape of the matrix V and the storage of the vectors which define
    the H(i) is best illustrated by the following example with n = 5 and
    k = 3.
    All elements including 0's and 1's are stored, unlike LAPACK.

        DIRECT = MagmaForward and         DIRECT = MagmaForward and
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = (  1  0  0 )                 V = (  1 v1 v1 v1 v1 )
                     ( v1  1  0 )                     (  0  1 v2 v2 v2 )
                     ( v1 v2  1 )                     (  0  0  1 v3 v3 )
                     ( v1 v2 v3 )
                     ( v1 v2 v3 )

        DIRECT = MagmaBackward and        DIRECT = MagmaBackward and
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = ( v1 v2 v3 )                 V = ( v1 v1  1  0  0 )
                     ( v1 v2 v3 )                     ( v2 v2 v2  1  0 )
                     (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
                     (  0  1 v3 )
                     (  0  0  1 )

    @ingroup magma_zaux3
    ********************************************************************/
template<typename Ty> magma_int_t
magma_larfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dV   , size_t dV_offset,    magma_int_t lddv,
    cl_mem dT   , size_t dT_offset,    magma_int_t lddt,
    cl_mem dC   , size_t dC_offset,    magma_int_t lddc,
    cl_mem dwork, size_t dwork_offset, magma_int_t ldwork,
    magma_queue_t queue )
{
    #define dV(i_,j_)  dV,     (dV_offset    + (i_) + (j_)*lddv)
    #define dT(i_,j_)  dT,     (dT_offset    + (i_) + (j_)*lddt)
    #define dC(i_,j_)  dC,     (dC_offset    + (i_) + (j_)*lddc)
    #define dwork(i_)  dwork,  (dwork_offset + (i_))

    static const Ty c_zero    = magma_zero<Ty>();
    static const Ty c_one     = magma_one<Ty>();
    static const Ty c_neg_one = magma_neg_one<Ty>();
    static const clblasTranspose transType = magma_is_real<Ty>() ? clblasTrans : clblasConjTrans;

    /* Check input arguments */
    magma_int_t info = 0;
    if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (k < 0) {
        info = -7;
    } else if ( ((storev == MagmaColumnwise) && (side == MagmaLeft) && lddv < std::max(1,m)) ||
                ((storev == MagmaColumnwise) && (side == MagmaRight) && lddv < std::max(1,n)) ||
                ((storev == MagmaRowwise) && lddv < k) ) {
        info = -9;
    } else if (lddt < k) {
        info = -11;
    } else if (lddc < std::max(1,m)) {
        info = -13;
    } else if ( ((side == MagmaLeft) && ldwork < std::max(1,n)) ||
                ((side == MagmaRight) && ldwork < std::max(1,m)) ) {
        info = -15;
    }
    if (info != 0) {
        //magma_xerbla( __func__, -(info) );
        return info;
    }

    /* Function Body */
    if (m <= 0 || n <= 0) {
        return info;
    }

    // opposite of trans
    clblasTranspose transt;
    clblasTranspose cltrans;
    if (trans == MagmaNoTrans) {
        transt = transType;
        cltrans = clblasNoTrans;
    }
    else {
        transt = clblasNoTrans;
        cltrans = transType;
    }

    // whether T is upper or lower triangular
    clblasUplo uplo;
    if (direct == MagmaForward)
        uplo = clblasUpper;
    else
        uplo = clblasLower;

    // whether V is stored transposed or not
    clblasTranspose notransV, transV;
    if (storev == MagmaColumnwise) {
        notransV = clblasNoTrans;
        transV   = transType;
    }
    else {
        notransV = transType;
        transV   = clblasNoTrans;
    }

    gemm_func<Ty> gpu_gemm;
    trmm_func<Ty> gpu_trmm;

    cl_event event = NULL;

    if ( side == MagmaLeft ) {
        // Form H C or H^H C
        // Comments assume H C. When forming H^H C, T gets transposed via transt.

        // W = C^H V
        gpu_gemm(clblasColumnMajor,
                 transType, notransV,
                 n, k, m,
                 c_one,
                 dC(0,0),  lddc,
                 dV(0,0),  lddv,
                 c_zero,
                 dwork(0), ldwork,
                 1, &queue, 0, nullptr, &event);

        // W = W T^H = C^H V T^H
        gpu_trmm(clblasColumnMajor,
                 clblasRight,
                 uplo, transt, clblasNonUnit,
                 n, k,
                 c_one,
                 dT(0,0) , lddt,
                 dwork(0), ldwork,
                 1, &queue, 0, nullptr, &event);

        // C = C - V W^H = C - V T V^H C = (I - V T V^H) C = H C
        gpu_gemm(clblasColumnMajor,
                 notransV, transType,
                 m, n, k,
                 c_neg_one,
                 dV(0,0),  lddv,
                 dwork(0), ldwork,
                 c_one,
                 dC(0,0),  lddc,
                 1, &queue, 0, nullptr, &event);
    }
    else {
        // Form C H or C H^H
        // Comments assume C H. When forming C H^H, T gets transposed via trans.

        // W = C V
        gpu_gemm(clblasColumnMajor,
                 clblasNoTrans, notransV,
                 m, k, n,
                 c_one,
                 dC(0,0),  lddc,
                 dV(0,0),  lddv,
                 c_zero,
                 dwork(0), ldwork,
                 1, &queue, 0, nullptr, &event);

        // W = W T = C V T
        gpu_trmm(clblasColumnMajor,
                 clblasRight, uplo,
                 cltrans,
                 clblasNonUnit,
                 m, k,
                 c_one,
                 dT(0,0),  lddt,
                 dwork(0), ldwork,
                 1, &queue, 0, nullptr, &event);

        // C = C - W V^H = C - C V T V^H = C (I - V T V^H) = C H
        gpu_gemm(clblasColumnMajor,
                 clblasNoTrans, transV,
                 m, n, k,
                 c_neg_one,
                 dwork(0), ldwork,
                 dV(0,0),  lddv,
                 c_one,
                 dC(0,0),  lddc,
                 1, &queue, 0, nullptr, &event);
    }

    return info;
} /* magma_zlarfb */

#define INSTANTIATE(T)                                          \
    template magma_int_t                                        \
    magma_larfb_gpu<T>(                                         \
        magma_side_t side, magma_trans_t trans,                 \
        magma_direct_t direct, magma_storev_t storev,           \
        magma_int_t m, magma_int_t n, magma_int_t k,            \
        cl_mem dV   , size_t dV_offset,    magma_int_t lddv,    \
        cl_mem dT   , size_t dT_offset,    magma_int_t lddt,    \
        cl_mem dC   , size_t dC_offset,    magma_int_t lddc,    \
        cl_mem dwork, size_t dwork_offset, magma_int_t ldwork,  \
        magma_queue_t queue );                                  \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
