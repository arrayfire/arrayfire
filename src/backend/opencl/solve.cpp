/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <solve.hpp>

#if defined(WITH_OPENCL_LINEAR_ALGEBRA)
#include <magma/magma.h>
#include <magma/magma_blas.h>
#include <magma/magma_data.h>
#include <magma/magma_helper.h>
#include <lu.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <blas.hpp>
#include <transpose.hpp>
#include <math.hpp>

#include <algorithm>
#include <string>

namespace opencl
{

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot,
                 const Array<T> &b, const af_mat_prop options)
{
    int N = A.dims()[0];
    int NRHS = b.dims()[1];

    std::vector<int> ipiv(N);
    copyData(&ipiv[0], pivot);

    Array< T > B = copyArray<T>(b);

    const cl::Buffer *A_buf = A.get();
    cl::Buffer *B_buf = B.get();

    int info = 0;
    magma_getrs_gpu<T>(MagmaNoTrans, N, NRHS,
                       (*A_buf)(), A.getOffset(), A.strides()[1],
                       &ipiv[0],
                       (*B_buf)(), B.getOffset(), B.strides()[1],
                       getQueue()(), &info);
    return B;
}

template<typename T>
Array<T> generalSolve(const Array<T> &a, const Array<T> &b)
{

    dim4 iDims = a.dims();
    int M = iDims[0];
    int N = iDims[1];
    int MN = std::min(M, N);
    std::vector<int> ipiv(MN);

    Array<T> A = copyArray<T>(a);
    Array<T> B = copyArray<T>(b);

    cl::Buffer *A_buf = A.get();
    int info = 0;
    magma_getrf_gpu<T>(M, N, (*A_buf)(), A.getOffset(), A.strides()[1],
                       &ipiv[0], getQueue()(), &info);

    cl::Buffer *B_buf = B.get();
    int K = B.dims()[1];
    magma_getrs_gpu<T>(MagmaNoTrans, M, K,
                       (*A_buf)(), A.getOffset(), A.strides()[1],
                       &ipiv[0],
                       (*B_buf)(), B.getOffset(), B.strides()[1],
                       getQueue()(), &info);
    return B;
}

template<typename T>
Array<T> leastSquares(const Array<T> &a, const Array<T> &b)
{
    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];
    int MN = std::min(M, N);

    Array<T> B = createEmptyArray<T>(dim4());
    gpu_blas_trsm_func<T> gpu_blas_trsm;

    cl_event event;
    cl_command_queue queue = getQueue()();

    if (M < N) {

#define UNMQR 0 // FIXME: UNMQR == 1 should be faster but does not work

        // Least squres for this case is solved using the following
        // solve(A, B) == matmul(Q, Xpad);
        // Where:
        // Xpad == pad(Xt, N - M, 1);
        // Xt   == tri_solve(R1, B);
        // R1   == R(seq(M), seq(M));
        // transpose(A) == matmul(Q, R);

        // QR is performed on the transpose of A
        Array<T> A = transpose<T>(a, true);

#if UNMQR
        B = padArray<T, T>(b, dim4(N, K), scalar<T>(0));
        B.resetDims(dim4(M, K));
#else
        B = copyArray<T>(b);
#endif

        int NB = magma_get_geqrf_nb<T>(A.dims()[1]);
        int NUM = (2*MN + ((M+31)/32)*32)*NB;
        Array<T> tmp = createEmptyArray<T>(dim4(NUM));

        std::vector<T> h_tau(MN);

        int info = 0;
        cl::Buffer *dA = A.get();
        cl::Buffer *dT = tmp.get();
        cl::Buffer *dB = B.get();

        magma_geqrf3_gpu<T>(A.dims()[0], A.dims()[1],
                           (*dA)(), A.getOffset(), A.strides()[1],
                           &h_tau[0], (*dT)(), tmp.getOffset(), getQueue()(), &info);

        A.resetDims(dim4(M, M));

        magmablas_swapdblk<T>(MN-1, NB,
                              (*dA)(), A.getOffset(), A.strides()[1], 1,
                              (*dT)(), tmp.getOffset() + MN * NB, NB, 0, queue);

        CLBLAS_CHECK(gpu_blas_trsm(
                         clblasLeft, clblasUpper,
                         clblasConjTrans, clblasNonUnit,
                         B.dims()[0], B.dims()[1],
                         scalar<T>(1),
                         (*dA)(), A.getOffset(), A.strides()[1],
                         (*dB)(), B.getOffset(), B.strides()[1],
                         1, &queue, 0, nullptr, &event));

        magmablas_swapdblk<T>(MN - 1, NB,
                              (*dT)(), tmp.getOffset() + MN * NB, NB, 0,
                              (*dA)(), A.getOffset(), A.strides()[1], 1, queue);

#if UNMQR
        int lwork = (B.dims()[0]-A.dims()[0]+NB)*(B.dims()[1]+2*NB);
        std::vector<T> h_work(lwork);
        B.resetDims(dim4(N, K));
        magma_unmqr_gpu<T>(MagmaLeft, MagmaNoTrans,
                           B.dims()[0], B.dims()[1], A.dims()[0],
                           (*dA)(), A.getOffset(), A.strides()[1],
                           &h_tau[0],
                           (*dB)(), B.getOffset(), B.strides()[1],
                           &h_work[0], lwork,
                           (*dT)(), tmp.getOffset(), NB, queue, &info);
#else
        A.resetDims(dim4(N, M));
        magma_ungqr_gpu<T>(A.dims()[0], A.dims()[1], std::min(M, N),
                           (*dA)(), A.getOffset(), A.strides()[1],
                           &h_tau[0],
                           (*dT)(), tmp.getOffset(), NB, queue, &info);

        B = matmul(A, B, AF_MAT_NONE, AF_MAT_NONE);
#endif
    } else if (M > N) {
        // Least squres for this case is solved using the following
        // solve(A, B) == tri_solve(R1, Bt);
        // Where:
        // R1 == R(seq(N), seq(N));
        // Bt == matmul(transpose(Q1), B);
        // Q1 == Q(span, seq(N));
        // A  == matmul(Q, R);

        Array<T> A = copyArray<T>(a);
        B = copyArray(b);

        int MN = std::min(M, N);
        int NB = magma_get_geqrf_nb<T>(M);

        int NUM = (2*MN + ((N+31)/32)*32)*NB;
        Array<T> tmp = createEmptyArray<T>(dim4(NUM));

        std::vector<T> h_tau(NUM);

        int info = 0;
        cl::Buffer *A_buf = A.get();
        cl::Buffer *B_buf = B.get();
        cl::Buffer *dT = tmp.get();

        magma_geqrf3_gpu<T>(M, N,
                           (*A_buf)(), A.getOffset(), A.strides()[1],
                           &h_tau[0], (*dT)(), tmp.getOffset(), getQueue()(), &info);

        int NRHS = B.dims()[1];
        int lhwork = (M - N + NB) * (NRHS + NB) + NRHS * NB;

        std::vector<T> h_work(lhwork);
        h_work[0] = scalar<T>(lhwork);

        magma_unmqr_gpu<T>(MagmaLeft, MagmaConjTrans,
                           M, NRHS, N,
                           (*A_buf)(), A.getOffset(), A.strides()[1],
                           &h_tau[0],
                           (*B_buf)(), B.getOffset(), B.strides()[1],
                           &h_work[0], lhwork,
                           (*dT)(), tmp.getOffset(), NB,
                           queue, &info);

        magmablas_swapdblk<T>(MN - 1, NB,
                              (*A_buf)(), A.getOffset(), A.strides()[1], 1,
                              (*dT)(), tmp.getOffset() + NB * MN,
                              NB, 0, queue);


        std::string pName = getPlatformName(getDevice());
        if(pName.find("NVIDIA") != std::string::npos)
        {
            Array<T> AT = transpose<T>(A, true);
            cl::Buffer* AT_buf = AT.get();
            CLBLAS_CHECK(gpu_blas_trsm(
                             clblasLeft, clblasLower, clblasConjTrans, clblasNonUnit,
                             N, NRHS, scalar<T>(1),
                             (*AT_buf)(), AT.getOffset(), AT.strides()[1],
                             (*B_buf)(), B.getOffset(), B.strides()[1],
                             1, &queue, 0, nullptr, &event));
        } else {
            CLBLAS_CHECK(gpu_blas_trsm(
                             clblasLeft, clblasUpper, clblasNoTrans, clblasNonUnit,
                             N, NRHS, scalar<T>(1),
                             (*A_buf)(), A.getOffset(), A.strides()[1],
                             (*B_buf)(), B.getOffset(), B.strides()[1],
                             1, &queue, 0, nullptr, &event));
        }
        B.resetDims(dim4(N, K));
    }

    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b, const af_mat_prop options)
{
    gpu_blas_trsm_func<T> gpu_blas_trsm;

    Array<T> B = copyArray<T>(b);

    int N = B.dims()[0];
    int NRHS = B.dims()[1];

    const cl::Buffer* A_buf = A.get();
    cl::Buffer* B_buf = B.get();

    cl_event event = 0;
    cl_command_queue queue = getQueue()();

    std::string pName = getPlatformName(getDevice());
    if(pName.find("NVIDIA") != std::string::npos && (options & AF_MAT_UPPER))
    {
        Array<T> AT = transpose<T>(A, true);

        cl::Buffer* AT_buf = AT.get();
        CLBLAS_CHECK(gpu_blas_trsm(
                         clblasLeft,
                         clblasLower,
                         clblasConjTrans,
                         options & AF_MAT_DIAG_UNIT ? clblasUnit : clblasNonUnit,
                         N, NRHS, scalar<T>(1),
                         (*AT_buf)(), AT.getOffset(), AT.strides()[1],
                         (*B_buf)(), B.getOffset(), B.strides()[1],
                         1, &queue, 0, nullptr, &event));
    } else {
        CLBLAS_CHECK(gpu_blas_trsm(
                         clblasLeft,
                         options & AF_MAT_LOWER ? clblasLower : clblasUpper,
                         clblasNoTrans,
                         options & AF_MAT_DIAG_UNIT ? clblasUnit : clblasNonUnit,
                         N, NRHS, scalar<T>(1),
                         (*A_buf)(), A.getOffset(), A.strides()[1],
                         (*B_buf)(), B.getOffset(), B.strides()[1],
                         1, &queue, 0, nullptr, &event));
    }

    return B;
}


template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_mat_prop options)
{
    try {
        initBlas();

        if (options & AF_MAT_UPPER ||
            options & AF_MAT_LOWER) {
            return triangleSolve<T>(a, b, options);
        }

        if(a.dims()[0] == a.dims()[1]) {
            return generalSolve<T>(a, b);
        } else {
            return leastSquares<T>(a, b);
        }
    } catch(cl::Error &err) {
        CL_TO_AF_ERROR(err);
    }
}

#define INSTANTIATE_SOLVE(T)                                            \
    template Array<T> solve<T>(const Array<T> &a, const Array<T> &b,    \
                               const af_mat_prop options);              \
    template Array<T> solveLU<T>(const Array<T> &A, const Array<int> &pivot, \
                                 const Array<T> &b, const af_mat_prop options); \

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)
}

#else

namespace opencl
{

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot,
                 const Array<T> &b, const af_mat_prop options)
{
    AF_ERROR("Linear Algebra is diabled on OpenCL",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_mat_prop options)
{
    AF_ERROR("Linear Algebra is diabled on OpenCL",
              AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                            \
    template Array<T> solve<T>(const Array<T> &a, const Array<T> &b,    \
                               const af_mat_prop options);              \
    template Array<T> solveLU<T>(const Array<T> &A, const Array<int> &pivot, \
                                 const Array<T> &b, const af_mat_prop options); \

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#endif
