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

template<typename T, typename BT>
using cptr_type     =   typename conditional<   is_complex<T>::value,
                                                const void *,
                                                const T*>::type;
template<typename T, typename BT>
using ptr_type     =    typename conditional<   is_complex<T>::value,
                                                void *,
                                                T*>::type;
template<typename T, typename BT>
using scale_type   =    typename conditional<   is_complex<T>::value,
                                                const void *,
                                                const T>::type;
template<typename T, typename BT>
using gemm_func_def = void (*)( const CBLAS_ORDER, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                                const int, const int, const int,
                                scale_type<T, BT>, cptr_type<T, BT>, const int,
                                cptr_type<T, BT>, const int,
                                scale_type<T, BT>, ptr_type<T, BT>, const int);

template<typename T, typename BT>
using gemv_func_def = void (*)( const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                                const int, const int,
                                scale_type<T, BT>, cptr_type<T, BT>, const int,
                                cptr_type<T, BT>, const int,
                                scale_type<T, BT>, ptr_type<T, BT>, const int);

#define BLAS_FUNC_DEF( FUNC )                                                      \
template<typename T, typename BT> FUNC##_func_def<T, BT> FUNC##_func();


#define BLAS_FUNC( FUNC, TYPE, BASE_TYPE, PREFIX )                                 \
template<> FUNC##_func_def<TYPE, BASE_TYPE>     FUNC##_func<TYPE, BASE_TYPE>()     \
{ return &cblas_##PREFIX##FUNC; }

BLAS_FUNC_DEF( gemm )
#ifdef OS_WIN
BLAS_FUNC(gemm , float   , float  , s)
BLAS_FUNC(gemm , double  , double , d)
BLAS_FUNC(gemm , cfloat  , float  , c)
BLAS_FUNC(gemm , cdouble , double , z)
#else
BLAS_FUNC(gemm , float   , float , s)
BLAS_FUNC(gemm , double  , double, d)
BLAS_FUNC(gemm , cfloat  ,   void, c)
BLAS_FUNC(gemm , cdouble ,   void, z)
#endif

BLAS_FUNC_DEF(gemv)
#ifdef OS_WIN
BLAS_FUNC(gemv , float   ,  float , s)
BLAS_FUNC(gemv , double  ,  double, d)
BLAS_FUNC(gemv , cfloat  ,  float , c)
BLAS_FUNC(gemv , cdouble ,  double, z)
#else
BLAS_FUNC(gemv , float   ,  float, s)
BLAS_FUNC(gemv , double  , double, d)
BLAS_FUNC(gemv , cfloat  ,   void, c)
BLAS_FUNC(gemv , cdouble ,   void, z)
#endif

template<typename T, typename BT, int value>
typename enable_if<is_floating_point<T>::value, scale_type<T,BT>>::type
getScale() { return T(value); }

template<typename T, typename BT, int value>
typename enable_if<is_complex<T>::value, scale_type<T,BT>>::type
getScale()
{
    static T val(value);
    return (const BT *)&val;
}

CBLAS_TRANSPOSE
toCblasTranspose(af_mat_prop opt)
{
    CBLAS_TRANSPOSE out = CblasNoTrans;
    switch(opt) {
        case AF_MAT_NONE        : out = CblasNoTrans;   break;
        case AF_MAT_TRANS           : out = CblasTrans;     break;
        case AF_MAT_CTRANS : out = CblasConjTrans; break;
        default                     : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

using namespace std;


#ifdef OS_WIN
#define BT af::dtype_traits<T>::base_type
#define REINTERPRET_CAST(PTR_TYPE, X) reinterpret_cast<PTR_TYPE>((X))
#else
template<typename T> struct cblas_types;

template<>
struct cblas_types<float> {
    typedef float base_type;
};

template<>
struct cblas_types<cfloat> {
    typedef void base_type;
};

template<>
struct cblas_types<double> {
    typedef double base_type;
};

template<>
struct cblas_types<cdouble> {
    typedef void base_type;
};
#define BT typename cblas_types<T>::base_type
#define REINTERPRET_CAST(PTR_TYPE, X) (X)
#endif

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
    auto alpha = getScale<T, BT, 1>();
    auto beta  = getScale<T, BT, 0>();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    if(rDims[bColDim] == 1) {
        N = lDims[aColDim];
        gemv_func<T, BT>()(
            CblasColMajor, lOpts,
            lDims[0], lDims[1],
            alpha, REINTERPRET_CAST(const BT*, lhs.get()), lStrides[1],
            REINTERPRET_CAST(const BT*, rhs.get()), rStrides[0],
            beta, REINTERPRET_CAST(BT*, out.get()), 1);
    } else {
        gemm_func<T, BT>()(
            CblasColMajor, lOpts, rOpts,
            M, N, K,
            alpha, REINTERPRET_CAST(const BT*, lhs.get()), lStrides[1],
            REINTERPRET_CAST(const BT*, rhs.get()), rStrides[1],
            beta, REINTERPRET_CAST(BT*, out.get()), out.dims()[0]);
    }

    return out;
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs,
             af_mat_prop optLhs, af_mat_prop optRhs)
{
    int N = lhs.dims()[0];

    T out = 0;
    const T *pL = lhs.get();
    const T *pR = rhs.get();

    for(int i = 0; i < N; i++)
        out += pL[i] * pR[i];

    return createValueArray(af::dim4(1), out);
}

#undef BT
#undef REINTEPRET_CAST

#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,  \
                                      af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                       \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                                   af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)

}
