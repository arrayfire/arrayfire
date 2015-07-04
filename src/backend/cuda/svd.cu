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

#if defined(WITH_CUDA_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>
#include "transpose.hpp"
#include <memory.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <err_common.hpp>

namespace cuda
{
    using cusolver::getDnHandle;

    template <typename T>
    struct gesvd_func_def_t {
        typedef cusolverStatus_t (*gesvd_func_def)(cusolverDnHandle_t, char, char, int,
                                                   int, T *, int, T *, T *, int, T *, int,
                                                   T *, int, T *, int *);
    };

    template<typename T>
    struct gesvd_buf_func_def_t {
        typedef cusolverStatus_t (*gesvd_buf_func_def)(cusolverDnHandle_t, int, int,
                                                       int *);
    };

#define SVD_FUNC_DEF(FUNC)                                                              \
    template <typename T>                                                               \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func();                       \
                                                                                        \
    template<typename T>                                                                \
    typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def                              \
    FUNC##_buf_func();

#define SVD_FUNC(FUNC, TYPE, PREFIX)                                                    \
    template <>                                                                         \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>()               \
    {                                                                                   \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) & cusolverDn##PREFIX##FUNC;   \
    }                                                                                   \
                                                                                        \
    template<> typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def                \
    FUNC##_buf_func<TYPE>()                                                             \
    {                                                                                   \
        return (FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def) &                     \
               cusolverDn##PREFIX##FUNC##_bufferSize;                                   \
    }

    SVD_FUNC_DEF(gesvd)
    SVD_FUNC(gesvd, float, S)
    SVD_FUNC(gesvd, double, D)
//SVD_FUNC(gesvd , cfloat , C)
//SVD_FUNC(gesvd , cdouble, Z)

    template <typename T>
    void svdInPlace(Array<T> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
    {
        dim4 iDims = in.dims();
        int M = iDims[0];
        int N = iDims[1];

        // cuSolver(cuda 7.0) doesn't have support for M<N
        bool flip_and_transpose = M < N;

        if (flip_and_transpose) {
            std::swap(M, N);
            std::swap(vt, u);
        }

        int lwork = 0;
        CUSOLVER_CHECK(gesvd_buf_func<T>()(getDnHandle(), M, N, &lwork));
        T *lWorkspace = memAlloc<T>(lwork);
        //complex numbers would need rWorkspace
        //T *rWorkspace = memAlloc<T>(lwork);

        int *info = memAlloc<int>(1);

        if (flip_and_transpose) {
            transpose_inplace(in, true);
            CUSOLVER_CHECK(gesvd_func<T>()(getDnHandle(), 'A', 'A', M, N, in.get(),
                                           M, s.get(), u.get(), M, vt.get(), N,
                                           lWorkspace, lwork, NULL, info));
            std::swap(u, vt);
            transpose_inplace(vt, true);
        } else {
            Array<T> inCopy = copyArray<T>(in);
            CUSOLVER_CHECK(gesvd_func<T>()(getDnHandle(), 'A', 'A', M, N, in.get(),
                                           M, s.get(), u.get(), M, vt.get(), N,
                                           lWorkspace, lwork, NULL, info));
        }
        memFree(info);
        memFree(lWorkspace);
        //memFree(rWorkspace);
    }

    template <typename T>
    void svd(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
    {
        Array<T> inCopy = copyArray<T>(in);
        svdInPlace(s, u, vt, inCopy);
    }

#define INSTANTIATE_SVD(T)                                                              \
    template void svd<T>(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in);   \
    template void svdInPlace<T>(Array<T> &s, Array<T> &u, Array<T> &vt, Array<T> &in);   \

    INSTANTIATE_SVD(float)
    //INSTANTIATE_SVD(cfloat)
    INSTANTIATE_SVD(double)
    //INSTANTIATE_SVD(cdouble)
}

#else
namespace cuda
{
    template <typename T>
    void svd(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
    {
        AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
                 AF_ERR_NOT_CONFIGURED);
    }

#define INSTANTIATE_SVD(T)                                                              \
    template void svd<T>(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in);   \

    INSTANTIATE_SVD(float)
    //INSTANTIATE_SVD(cfloat)
    INSTANTIATE_SVD(double)
    //INSTANTIATE_SVD(cdouble)
}
#endif
