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

#if defined(WITH_CPU_LINEAR_ALGEBRA)
#include <lapack_helper.hpp>
#include <copy.hpp>

namespace cpu
{

#define SVD_FUNC_DEF( FUNC )                                            \
    template<typename T,typename Tr> svd_func_def<T, Tr> svd_func();

#define SVD_FUNC( FUNC, T, Tr, PREFIX )                     \
    template<> svd_func_def<T, Tr>     svd_func<T, Tr>()    \
    { return & LAPACK_NAME(PREFIX##FUNC); }

#if defined(USE_MKL) || defined(__APPLE__)

    template<typename T, typename Tr>
    using svd_func_def = int (*)(ORDER_TYPE,
                                 char jobz,
                                 int m, int n,
                                 T* in, int ldin,
                                 Tr* s,
                                 T* u, int ldu,
                                 T* vt, int ldvt);

    SVD_FUNC_DEF( gesdd )
    SVD_FUNC(gesdd, float  , float , s)
    SVD_FUNC(gesdd, double , double, d)
    SVD_FUNC(gesdd, cfloat , float , c)
    SVD_FUNC(gesdd, cdouble, double, z)

#else   // Atlas causes memory freeing issues with using gesdd

    template<typename T, typename Tr>
    using svd_func_def = int (*)(ORDER_TYPE,
                                 char jobu, char jobvt,
                                 int m, int n,
                                 T* in, int ldin,
                                 Tr* s,
                                 T* u, int ldu,
                                 T* vt, int ldvt,
                                 Tr *superb);

    SVD_FUNC_DEF( gesvd )
    SVD_FUNC(gesvd, float  , float , s)
    SVD_FUNC(gesvd, double , double, d)
    SVD_FUNC(gesvd, cfloat , float , c)
    SVD_FUNC(gesvd, cdouble, double, z)

#endif

    template <typename T, typename Tr>
    void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
    {
        dim4 iDims = in.dims();
        int M = iDims[0];
        int N = iDims[1];

#if defined(USE_MKL) || defined(__APPLE__)
        svd_func<T, Tr>()(AF_LAPACK_COL_MAJOR, 'A', M, N, in.get(), in.strides()[1],
                          s.get(), u.get(), u.strides()[1], vt.get(), vt.strides()[1]);
#else
        std::vector<Tr> superb(std::min(M, N));
        svd_func<T, Tr>()(AF_LAPACK_COL_MAJOR, 'A', 'A', M, N, in.get(), in.strides()[1],
                          s.get(), u.get(), u.strides()[1], vt.get(), vt.strides()[1], &superb[0]);
#endif
    }

    template <typename T, typename Tr>
    void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
    {
        Array<T> in_copy = copyArray<T>(in);
        svdInPlace(s, u, vt, in_copy);
    }
}

#else

namespace cpu
{
    template <typename T, typename Tr>
    void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
    {
        AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
    }

    template <typename T, typename Tr>
    void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
    {
        AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
    }
}

#endif

namespace cpu {

#define INSTANTIATE_SVD(T, Tr)                                          \
    template void svd<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, const Array<T> &in); \
    template void svdInPlace<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, Array<T> &in);

    INSTANTIATE_SVD(float  , float )
    INSTANTIATE_SVD(double , double)
    INSTANTIATE_SVD(cfloat , float )
    INSTANTIATE_SVD(cdouble, double)
}
