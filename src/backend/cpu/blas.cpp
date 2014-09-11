#include <blas.hpp>
#include <af/dim4.hpp>
#include <handle.hpp>
#include <iostream>
#include <cassert>

#include <cassert>

namespace cpu
{

    using std::add_const;
    using std::add_pointer;
    using std::enable_if;
    using std::is_floating_point;
    using std::remove_const;
    using std::conditional;

template<typename T>
using cptr_type     =   typename conditional<   is_complex<T>::value,
                                                const void *,
                                                const T*>::type;
template<typename T>
using ptr_type     =    typename conditional<   is_complex<T>::value,
                                                void *,
                                                T*>::type;
template<typename T>
using scale_type     =  typename conditional<   is_complex<T>::value,
                                                const void *,
                                                T>::type;
template<typename T>
using gemm_func_def = void (*)( const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE,
                                const int, const int, const int,
                                scale_type<T>, cptr_type<T>, const int,
                                cptr_type<T>, const int,
                                scale_type<T>, ptr_type<T>, const int);

template<typename T>
using gemv_func_def = void (*)( const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE,
                                const int, const int,
                                scale_type<T>, cptr_type<T>, const int,
                                cptr_type<T>, const int,
                                scale_type<T>, ptr_type<T>, const int);

template<typename T>
using dot_func_def = T (*) (    const int,
                                cptr_type<T>,
                                const int,
                                cptr_type<T>,
                                const int);

#define BLAS_FUNC_DEF( FUNC )                                                                   \
template<typename T> FUNC##_func_def<T> FUNC##_func();


#define BLAS_FUNC( FUNC, TYPE, PREFIX )                                                         \
template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()     { return &cblas_##PREFIX##FUNC; }

BLAS_FUNC_DEF( gemm )
BLAS_FUNC(gemm, float,      s)
BLAS_FUNC(gemm, double,     d)
BLAS_FUNC(gemm, cfloat,     c)
BLAS_FUNC(gemm, cdouble,    z)

BLAS_FUNC_DEF( gemv )
BLAS_FUNC(gemv, float,      s)
BLAS_FUNC(gemv, double,     d)
BLAS_FUNC(gemv, cfloat,     c)
BLAS_FUNC(gemv, cdouble,    z)

BLAS_FUNC_DEF( dot )
BLAS_FUNC(dot, float,       s)
BLAS_FUNC(dot, double,      d)

template<typename T, int value>
typename enable_if<is_floating_point<T>::value, scale_type<T>>::type
getScale() { return T(value); }

template<typename T, int value>
typename enable_if<is_complex<T>::value, scale_type<T>>::type
getScale()
{
    static T val(value);
    return &val;
}

CBLAS_TRANSPOSE
toCblasTranspose(af_blas_transpose opt)
{
    CBLAS_TRANSPOSE out;
    switch(opt) {
        case AF_NO_TRANSPOSE        : out = CblasNoTrans;   break;
        case AF_TRANSPOSE           : out = CblasTrans;     break;
        case AF_CONJUGATE_TRANSPOSE : out = CblasConjTrans; break;
        default                     : assert( "INVALID af_blas_transpose" && 1!=1);
    }
    return out;
}

#include <iostream>
using namespace std;

template<typename T>
Array<T>* matmul(const Array<T> &lhs, const Array<T> &rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    CBLAS_TRANSPOSE lOpts = toCblasTranspose(optLhs);
    CBLAS_TRANSPOSE rOpts = toCblasTranspose(optRhs);

    int aRowDim = (lOpts == CblasNoTrans) ? 0 : 1;
    int aColDim = (lOpts == CblasNoTrans) ? 1 : 0;
    int bRowDim = (rOpts == CblasNoTrans) ? 0 : 1;
    int bColDim = (rOpts == CblasNoTrans) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[aRowDim];
    int N = rDims[bColDim];
    int K = lDims[aColDim];

    assert(lDims[aColDim] == rDims[bRowDim]);

    //FIXME: Leaks on errors.
    Array<T> *out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    auto alpha = getScale<T, 1>();
    auto beta  = getScale<T, 0>();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    if(rDims[bColDim] == 1) {
        N = lDims[aColDim];
        gemv_func<T>()(
            CblasColMajor, lOpts,
            M, N,
            alpha,  lhs.get(),   lStrides[1],
                    rhs.get(),   rStrides[0],
            beta ,  out->get(),           1);
    } else {
        gemm_func<T>()(
            CblasColMajor, lOpts, rOpts,
            M, N, K,
            alpha,  lhs.get(),   lStrides[1],
                    rhs.get(),   rStrides[1],
            beta ,  out->get(),  out->dims()[0]);
    }

    return out;
}

template<typename T>
Array<T>* dot(const Array<T> &lhs, const Array<T> &rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    assert(lhs.dims()[0] == rhs.dims()[0]);
    int N = lhs.dims()[0];

    T out = dot_func<T>()(  N,
                            lhs.get(), lhs.strides()[0],
                            rhs.get(), rhs.strides()[0]
                            );

    return createValueArray(af::dim4(1), out);
}

#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE>* matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,  \
                    af_blas_transpose optLhs, af_blas_transpose optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                       \
    template Array<TYPE>* dot<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                    af_blas_transpose optLhs, af_blas_transpose optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)

}
