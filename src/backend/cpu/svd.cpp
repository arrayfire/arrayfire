/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <svd.hpp>
#include <err_common.hpp>

#include <err_cpu.hpp>

#define INSTANTIATE_SVD(T)                                                               \
    template void svd<T>(Array<T> & s, Array<T> & u, Array<T> & vt, const Array<T> &in); \
    template void svdInPlace<T>(Array<T> & s, Array<T> & u, Array<T> & vt, Array<T> &in);

#if defined(WITH_CPU_LINEAR_ALGEBRA)
#include <lapack_helper.hpp>
#include <copy.hpp>

namespace cpu
{

    #define SVD_FUNC_DEF( FUNC )                                     \
    template<typename T> FUNC##_func_def<T> FUNC##_func();

    #define SVD_FUNC( FUNC, TYPE, PREFIX )                           \
    template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()         \
    { return & LAPACK_NAME(PREFIX##FUNC); }

    template<typename T>
    using gesdd_func_def = int (*)(ORDER_TYPE, char jobz, int m, int n, T* in,
                                     int ldin, T* s, T* u, int ldu,
                                     T* vt, int ldvt);

    SVD_FUNC_DEF( gesdd )
    SVD_FUNC(gesdd, float, s)
    SVD_FUNC(gesdd, double, d)

    template <typename T>
    void svdInPlace(Array<T> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
    {
        dim4 iDims = in.dims();
        int M = iDims[0];
        int N = iDims[1];

        gesdd_func<T>()(AF_LAPACK_COL_MAJOR, 'A', M, N, in.get(), in.strides()[1],
                        s.get(), u.get(), u.strides()[1], vt.get(), vt.strides()[1]);
    }

    template <typename T>
    void svd(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
    {
        Array<T> in_copy = copyArray<T>(in);
        svdInPlace(s, u, vt, in_copy);
    }
}

#else

namespace cpu
{
    template <typename T>
    void svd(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
    {
        AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
    }

    template <typename T>
    void svdInPlace(Array<T> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
    {
        AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
    }
}

#endif

namespace cpu {
    INSTANTIATE_SVD(float)
    INSTANTIATE_SVD(double)
}
