/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <af/dim4.hpp>
#include <handle.hpp>
#include <cassert>
#include <err_cpu.hpp>
#include <err_common.hpp>

namespace cpu
{

    using std::add_const;
    using std::add_pointer;
    using std::enable_if;
    using std::is_floating_point;
    using std::remove_const;
    using std::conditional;

  // Some implementations of BLAS require void* for complex pointers while
  // others use float*/double*
#if (defined(OS_WIN) && !defined(USE_MKL)) || defined(IS_OPENBLAS)
  static const bool cplx_void_ptr = false;
#else
  static const bool cplx_void_ptr = true;
#endif

template<typename T, class Enable = void>
struct blas_base {
    using type = typename dtype_traits<T>::base_type;
};

template<typename T>
struct blas_base <T, typename enable_if<is_complex<T>::value && cplx_void_ptr>::type> {
    using type = void;
};


template<typename T>
using cptr_type     =   typename conditional<   is_complex<T>::value,
                                                const typename blas_base<T>::type *,
                                                const T*>::type;
template<typename T>
using ptr_type     =    typename conditional<   is_complex<T>::value,
                                                typename blas_base<T>::type *,
                                                T*>::type;
template<typename T>
using scale_type   =    typename conditional<   is_complex<T>::value,
                                                const typename blas_base<T>::type *,
                                                const T>::type;

template<typename T>
using gemm_func_def = void (*)( const CBLAS_ORDER, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                                const blasint, const blasint, const blasint,
                                scale_type<T>, cptr_type<T>, const blasint,
                                cptr_type<T>, const blasint,
                                scale_type<T>, ptr_type<T>, const blasint);

template<typename T>
using gemv_func_def = void (*)( const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                                const blasint, const blasint,
                                scale_type<T>, cptr_type<T>, const blasint,
                                cptr_type<T>, const blasint,
                                scale_type<T>, ptr_type<T>, const blasint);

#define BLAS_FUNC_DEF( FUNC )                           \
template<typename T> FUNC##_func_def<T> FUNC##_func();

#define BLAS_FUNC( FUNC, TYPE, PREFIX )                 \
  template<> FUNC##_func_def<TYPE> FUNC##_func<TYPE>()  \
{ return &cblas_##PREFIX##FUNC; }

BLAS_FUNC_DEF( gemm )
BLAS_FUNC(gemm , float   , s)
BLAS_FUNC(gemm , double  , d)
BLAS_FUNC(gemm , cfloat  , c)
BLAS_FUNC(gemm , cdouble , z)

BLAS_FUNC_DEF(gemv)
BLAS_FUNC(gemv , float   , s)
BLAS_FUNC(gemv , double  , d)
BLAS_FUNC(gemv , cfloat  , c)
BLAS_FUNC(gemv , cdouble , z)

template<typename T, int value>
typename enable_if<is_floating_point<T>::value, scale_type<T>>::type
getScale() { return T(value); }

template<typename T, int value>
typename enable_if<is_complex<T>::value, scale_type<T>>::type
getScale()
{
    static T val(value);
    return (const typename blas_base<T>::type *)&val;
}

CBLAS_TRANSPOSE
toCblasTranspose(af_mat_prop opt)
{
    CBLAS_TRANSPOSE out = CblasNoTrans;
    switch(opt) {
        case AF_MAT_NONE        : out = CblasNoTrans;   break;
        case AF_MAT_TRANS       : out = CblasTrans;     break;
        case AF_MAT_CTRANS      : out = CblasConjTrans; break;
        default                 : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    CBLAS_TRANSPOSE lOpts = toCblasTranspose(optLhs);
    CBLAS_TRANSPOSE rOpts = toCblasTranspose(optRhs);

    int aRowDim = (lOpts == CblasNoTrans) ? 0 : 1;
    int aColDim = (lOpts == CblasNoTrans) ? 1 : 0;
    int bColDim = (rOpts == CblasNoTrans) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[aRowDim];
    int N = rDims[bColDim];
    int K = lDims[aColDim];

    //FIXME: Leaks on errors.
    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    auto alpha = getScale<T, 1>();
    auto beta  = getScale<T, 0>();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    using BT  =       typename blas_base<T>::type;
    using CBT = const typename blas_base<T>::type;

    if(rDims[bColDim] == 1) {
        N = lDims[aColDim];
        gemv_func<T>()(
            CblasColMajor, lOpts,
            lDims[0], lDims[1],
            alpha,
            reinterpret_cast<CBT*>(lhs.get()), lStrides[1],
            reinterpret_cast<CBT*>(rhs.get()), rStrides[0],
            beta,
            reinterpret_cast<BT*>(out.get()), 1);
    } else {
        gemm_func<T>()(
            CblasColMajor, lOpts, rOpts,
            M, N, K,
            alpha,
            reinterpret_cast<CBT*>(lhs.get()), lStrides[1],
            reinterpret_cast<CBT*>(rhs.get()), rStrides[1],
            beta,
            reinterpret_cast<BT*>(out.get()), out.dims()[0]);
    }

    return out;
}

template<typename T> T
conj(T  x) { return x; }

template<> cfloat  conj<cfloat> (cfloat  c) { return std::conj(c); }
template<> cdouble conj<cdouble>(cdouble c) { return std::conj(c); }

template<typename T, bool conjugate, bool both_conjugate>
Array<T> dot_(const Array<T> &lhs, const Array<T> &rhs,
              af_mat_prop optLhs, af_mat_prop optRhs)
{
    int N = lhs.dims()[0];

    T out = 0;
    const T *pL = lhs.get();
    const T *pR = rhs.get();

    for(int i = 0; i < N; i++)
        out += (conjugate ? cpu::conj(pL[i]) : pL[i]) * pR[i];

    if(both_conjugate) out = cpu::conj(out);

    return createValueArray(af::dim4(1), out);
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs,
             af_mat_prop optLhs, af_mat_prop optRhs)
{
    if(optLhs == AF_MAT_CONJ && optRhs == AF_MAT_CONJ) {
        return dot_<T, false, true>(lhs, rhs, optLhs, optRhs);
    } else if (optLhs == AF_MAT_CONJ && optRhs == AF_MAT_NONE) {
        return dot_<T, true, false>(lhs, rhs, optLhs, optRhs);
    } else if (optLhs == AF_MAT_NONE && optRhs == AF_MAT_CONJ) {
        return dot_<T, true, false>(rhs, lhs, optRhs, optLhs);
    } else {
        return dot_<T, false, false>(lhs, rhs, optLhs, optRhs);
    }
}

#undef BT
#undef REINTEPRET_CAST

#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,   \
                                      af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                               \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,          \
                                   af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)

}
