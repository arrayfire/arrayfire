/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <svd.hpp>
#include <err_common.hpp>

#include <cusolverDnManager.hpp>
#include "transpose.hpp"
#include <memory.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <err_common.hpp>

namespace cuda
{

#if defined(WITH_CUDA_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>

    using cusolver::getDnHandle;

    template<typename T>
    cusolverStatus_t gesvd_buf_func(cusolverDnHandle_t handle, int m, int n, int *Lwork)
    {
        return CUSOLVER_STATUS_ARCH_MISMATCH;
    }

    template<typename T, typename Tr>
    cusolverStatus_t gesvd_func(cusolverDnHandle_t handle, char jobu, char jobvt,
                                int m, int n,
                                T *A, int lda,
                                Tr *S,
                                T *U, int ldu,
                                T *VT, int ldvt,
                                T *Work, int Lwork,
                                Tr *rwork, int *devInfo)
    {
        return CUSOLVER_STATUS_ARCH_MISMATCH;
    }

#define SVD_SPECIALIZE(T, Tr, X)                                        \
    template<> cusolverStatus_t                                         \
    gesvd_buf_func<T>(cusolverDnHandle_t handle,                        \
                      int m, int n, int *Lwork)                         \
    {                                                                   \
        return cusolverDn##X##gesvd_bufferSize(handle, m, n, Lwork);    \
    }                                                                   \

SVD_SPECIALIZE(float  , float , S);
SVD_SPECIALIZE(double , double, D);
SVD_SPECIALIZE(cfloat , float , C);
SVD_SPECIALIZE(cdouble, double, Z);

#undef SVD_SPECIALIZE

#define SVD_SPECIALIZE(T, Tr, X)                                        \
    template<> cusolverStatus_t                                         \
    gesvd_func<T, Tr>(cusolverDnHandle_t handle,                        \
                      char jobu, char jobvt,                            \
                      int m, int n,                                     \
                      T *A, int lda,                                    \
                      Tr *S,                                            \
                      T *U, int ldu,                                    \
                      T *VT, int ldvt,                                  \
                      T *Work, int Lwork,                               \
                      Tr *rwork, int *devInfo)                          \
    {                                                                   \
        return cusolverDn##X##gesvd(handle, jobu, jobvt,                \
                                    m, n, A, lda, S, U, ldu, VT, ldvt,  \
                                    Work, Lwork, rwork, devInfo);       \
    }                                                                   \

SVD_SPECIALIZE(float  , float , S);
SVD_SPECIALIZE(double , double, D);
SVD_SPECIALIZE(cfloat , float , C);
SVD_SPECIALIZE(cdouble, double, Z);

    template <typename T, typename Tr>
    void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
    {
        dim4 iDims = in.dims();
        int M = iDims[0];
        int N = iDims[1];

        int lwork = 0;

        CUSOLVER_CHECK(gesvd_buf_func<T>(getDnHandle(), M, N, &lwork));

        T  *lWorkspace = memAlloc<T >(lwork);
        Tr *rWorkspace = memAlloc<Tr>(5 * std::min(M, N));

        int *info = memAlloc<int>(1);

        gesvd_func<T, Tr>(getDnHandle(), 'A', 'A', M, N, in.get(),
                          M, s.get(), u.get(), M, vt.get(), N,
                          lWorkspace, lwork, rWorkspace, info);

        memFree(info);
        memFree(lWorkspace);
        memFree(rWorkspace);
    }

    template <typename T, typename Tr>
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
            transpose_inplace(vt, true);
            transpose_inplace(u, true);
        }
    }

#else

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
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
