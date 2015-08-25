/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <svd.hpp>               // opencl backend function header
#include <err_opencl.hpp>        // error check functions and Macros
#include <reduce.hpp>
#include <copy.hpp>
#include <blas.hpp>
#include <transpose.hpp>

#include <magma/magma.h>
#include <magma/magma_cpu_lapack.h>
#include <magma/magma_helper.h>

#if defined(WITH_OPENCL_LINEAR_ALGEBRA)
namespace opencl
{

template<typename Tr>
Tr calc_scale(Tr From, Tr To)
{
    //FIXME: I am not sure this is correct, removing this for now
#if 0
    //http://www.netlib.org/lapack/explore-3.1.1-html/dlascl.f.html
    cpu_lapack_lamch_func<Tr> cpu_lapack_lamch;

    Tr S = cpu_lapack_lamch('S');
    Tr B = 1.0 / S;

    Tr FromCopy = From, ToCopy = To;

    Tr Mul = 1;

    while (true) {
        Tr From1 = FromCopy * S, To1 = ToCopy / B;
        if (std::abs(From1) > std::abs(ToCopy) && ToCopy != 0) {
            Mul *= S;
            FromCopy = From1;
        } else if (std::abs(To1) > std::abs(FromCopy)) {
            Mul *= B;
            ToCopy = To1;
        } else {
            Mul *= (ToCopy) / (FromCopy);
            break;
        }
    }

    return Mul;
#else
    return To / From;
#endif
}

template<typename T, typename Tr>
void svd(Array<T > &arrU,
         Array<Tr> &arrS,
         Array<T > &arrVT,
         Array<T > &arrA,
         bool want_vectors=true)
{

    dim4 idims = arrA.dims();
    dim4 istrides = arrA.strides();

    const int m = (int)idims[0];
    const int n = (int)idims[1];
    const int ldda = (int)istrides[1];
    const int lda = m;
    const int min_mn = std::min(m, n);
    const int ldu = m;
    const int ldvt = n;

    const int nb   = magma_get_gebrd_nb<T>(n);
    const int lwork = (m + n) * nb;

    cpu_lapack_lacpy_func<T> cpu_lapack_lacpy;
    cpu_lapack_bdsqr_work_func<T> cpu_lapack_bdsqr_work;
    cpu_lapack_ungbr_work_func<T> cpu_lapack_ungbr_work;
    cpu_lapack_lamch_func<Tr> cpu_lapack_lamch;

    // Get machine constants
    static const double eps = cpu_lapack_lamch('P');
    static const double smlnum = std::sqrt(cpu_lapack_lamch('S')) / eps;
    static const double bignum = 1. / smlnum;

    Tr anrm = abs(reduce_all<af_max_t, T, T>(arrA));

    T scale = scalar<T>(1);
    static const int ione  = 1;
    static const int izero = 0;


    bool iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        scale = scalar<T>(calc_scale<Tr>(anrm, smlnum));
    } else if (anrm > bignum) {
        iscl = 1;
        scale = scalar<T>(calc_scale<Tr>(anrm, bignum));
    }

    if (iscl == 1) {
        multiply_inplace(arrA, abs(scale));
    }

    int nru = 0;
    int ncvt = 0;

    std::vector<T> A(m * n);
    std::vector<T> tauq(min_mn), taup(min_mn);
    std::vector<T> work(lwork);
    std::vector<Tr> s0(min_mn), s1(min_mn - 1);
    std::vector<Tr> rwork(5 * min_mn);

    int info = 0;

    copyData(&A[0], arrA);


    // Bidiagonalize A
    // (CWorkspace: need 2*N + M, prefer 2*N + (M + N)*NB)
    // (RWorkspace: need N)
    magma_gebrd_hybrid<T>(m, n,
                          &A[0], lda,
                          (*arrA.get())(), arrA.getOffset(), ldda,
                          (void *)&s0[0], (void *)&s1[0],
                          &tauq[0], &taup[0],
                          &work[0], lwork,
                          getQueue()(), &info, false);

    std::vector<T> U(1), VT(1);
    std::vector<T> cdummy(1);

    if (want_vectors) {

        U = std::vector<T>(m * m);
        VT = std::vector<T>(n * n);

        // If left singular vectors desired in U, copy result to U
        // and generate left bidiagonalizing vectors in U
        // (CWorkspace: need 2*N + NCU, prefer 2*N + NCU*NB)
        // (RWorkspace: 0)
        LAPACKE_CHECK(cpu_lapack_lacpy('L', m, n, &A[0], lda, &U[0], ldu));

        int ncu = m;
        LAPACKE_CHECK(cpu_lapack_ungbr_work('Q', m, ncu, n, &U[0], ldu, &tauq[0], &work[0], lwork));

        // If right singular vectors desired in VT, copy result to
        // VT and generate right bidiagonalizing vectors in VT
        // (CWorkspace: need 3*N-1, prefer 2*N + (N-1)*NB)
        // (RWorkspace: 0)
        LAPACKE_CHECK(cpu_lapack_lacpy('U', n, n, &A[0], lda, &VT[0], ldvt));
        LAPACKE_CHECK(cpu_lapack_ungbr_work('P', n, n, n, &VT[0], ldvt, &taup[0], &work[0], lwork));

        nru = m;
        ncvt = n;
    }

    // Perform bidiagonal QR iteration, if desired, computing
    // left singular vectors in U and computing right singular
    // vectors in VT
    // (CWorkspace: need 0)
    // (RWorkspace: need BDSPAC)
    LAPACKE_CHECK(cpu_lapack_bdsqr_work('U', n, ncvt, nru, izero,
                                        &s0[0], &s1[0], &VT[0], ldvt, &U[0], ldu,
                                        &cdummy[0], ione, &rwork[0]));


    if (want_vectors) {
        writeHostDataArray(arrU, &U[0], arrU.elements() * sizeof(T));
        writeHostDataArray(arrVT, &VT[0], arrVT.elements() * sizeof(T));
    }

    writeHostDataArray(arrS, &s0[0], arrS.elements() * sizeof(Tr));

    if (iscl == 1) {
        Tr rscale = scalar<Tr>(1);
        if (anrm > bignum) {
            rscale = calc_scale<Tr>(bignum, anrm);
        } else if (anrm < smlnum) {
            rscale = calc_scale<Tr>(smlnum, anrm);
        }
        multiply_inplace(arrS, rscale);
    }
}


template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
{
    initBlas();
    svd<T, Tr>(u, s, vt, in, true);
}

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    if (M >= N) {
        Array<T> in_copy = copyArray(in);
        svdInPlace(s, u, vt, in_copy);
    } else {
        Array<T> in_trans = transpose(in, true);
        svdInPlace(s, vt, u, in_trans);
        transpose_inplace(u, true);
        transpose_inplace(vt, true);
    }
}

#else

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}
#endif

#define INSTANTIATE(T, Tr)                                              \
    template void svd<T, Tr>(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in); \
    template void svdInPlace<T, Tr>(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}
