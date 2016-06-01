/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>
#include <mkl_spblas.h>

#include <stdexcept>
#include <string>
#include <cassert>

#include <af/dim4.hpp>
#include <complex.hpp>
#include <handle.hpp>
#include <err_common.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace cpu
{

using namespace common;

using std::add_const;
using std::add_pointer;
using std::enable_if;
using std::is_floating_point;
using std::remove_const;
using std::conditional;
using std::is_same;

template<typename T, class Enable = void>
struct blas_base {
    using type = T;
};

template<typename T>
struct blas_base <T, typename enable_if<is_complex<T>::value>::type> {
    using type = typename conditional<is_same<T, cdouble>::value,
                                      sp_cdouble, sp_cfloat>
                                     ::type;
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
                                                const T *>::type;
// MKL
// void mkl_zcsrmm (const char *transa ,
//                  const MKL_INT *m , const MKL_INT *n , const MKL_INT *k ,
//                  const MKL_Complex16 *alpha , const char *matdescra ,
//                  const MKL_Complex16 *val , const MKL_INT *indx ,
//                  const MKL_INT *pntrb , const MKL_INT *pntre ,
//                  const MKL_Complex16 *b , const MKL_INT *ldb ,
//                  const MKL_Complex16 *beta ,
//                  MKL_Complex16 *c , const MKL_INT *ldc );
//
// void mkl_zcsrmv (const char *transa ,
//                  const MKL_INT *m , const MKL_INT *k ,
//                  const MKL_Complex16 *alpha , const char *matdescra ,
//                  const MKL_Complex16 *val , const MKL_INT *indx ,
//                  const MKL_INT *pntrb , const MKL_INT *pntre ,
//                  const MKL_Complex16 *x ,
//                  const MKL_Complex16 *beta ,
//                  MKL_Complex16 *y );
//

template<typename T>
using csrmm_func_def = void (*)( const sp_op_t *,
                                 const int *, const int *, const int *,
                                 const scale_type<T>, const sp_op_t *,
                                 cptr_type<T>, const int *,
                                 const int *, const int *,
                                 cptr_type<T>, const int *,
                                 scale_type<T>,
                                 ptr_type<T>, const int *);

template<typename T>
using csrmv_func_def = void (*)( const sp_op_t *,
                                 const int *, const int *,
                                 const scale_type<T>, const sp_op_t *,
                                 cptr_type<T>, const int *,
                                 const int *, const int *,
                                 cptr_type<T>,
                                 scale_type<T>,
                                 ptr_type<T>);

#define SPARSE_FUNC_DEF( FUNC )                         \
template<typename T> FUNC##_func_def<T> FUNC##_func();

#define SPARSE_FUNC( FUNC, TYPE, PREFIX )               \
  template<> FUNC##_func_def<TYPE> FUNC##_func<TYPE>()  \
{ return &mkl_##PREFIX##FUNC; }

SPARSE_FUNC_DEF( csrmm )
SPARSE_FUNC(csrmm , float   , s)
SPARSE_FUNC(csrmm , double  , d)
SPARSE_FUNC(csrmm , cfloat  , c)
SPARSE_FUNC(csrmm , cdouble , z)

SPARSE_FUNC_DEF( csrmv )
SPARSE_FUNC(csrmv , float   , s)
SPARSE_FUNC(csrmv , double  , d)
SPARSE_FUNC(csrmv , cfloat  , c)
SPARSE_FUNC(csrmv , cdouble , z)

template<typename T, int value>
scale_type<T> getScale()
{
    static T val(value);
    return (const typename blas_base<T>::type*)&val;
}

sp_op_t
toSparseTranspose(af_mat_prop opt)
{
    sp_op_t out = 'N';
    switch(opt) {
        case AF_MAT_NONE        : out = 'N';    break;
        case AF_MAT_TRANS       : out = 'T';    break;
        case AF_MAT_CTRANS      : out = 'C';    break;
        default                 : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    // MKL: CSRMM Does not support optRhs

    lhs.eval();
    rhs.eval();

    // Similar Operations to GEMM
    sp_op_t lOpts = toSparseTranspose(optLhs);

    int lRowDim = (lOpts == 'N') ? 0 : 1;
    int lColDim = (lOpts == 'N') ? 1 : 0;
    static const int rColDim = 1; //Unsupported : (rOpts == 'N;) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[lRowDim];
    int N = rDims[rColDim];
    int K = lDims[lColDim];

    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));
    out.eval();

    auto func = [=] (Array<T> output, const SparseArray<T> left, const Array<T> right) {
        // Mat Descr
        // When 0 is 'G', 1, 2 are ignored
        // 4 and 5 are unused
        static const sp_op_t descra[] = {'G', '0', '0', 'C', '0', '0'};

        auto alpha = getScale<T, 1>();
        auto beta  = getScale<T, 0>();

        int ldb = rhs.strides()[1];
        int ldc = out.strides()[1];

        Array<T  > values = lhs.getValues();
        Array<int> rowIdx = lhs.getRowIdx();
        Array<int> colIdx = lhs.getColIdx();

        const int *pB = rowIdx.get();
        const int *pE = rowIdx.get() + 1;

        if(rDims[rColDim] == 1) {
            csrmv_func<T>()(
                &lOpts, &M, &K,
                reinterpret_cast<scale_type<T>>(&alpha), descra,
                reinterpret_cast<cptr_type<T>>(values.get()),
                reinterpret_cast<const int*>(colIdx.get()),
                pB, pE,
                reinterpret_cast<cptr_type<T>>(rhs.get()),
                reinterpret_cast<scale_type<T>>(&beta),
                reinterpret_cast<ptr_type<T>>(const_cast<T*>(out.get())));
        } else {
            csrmm_func<T>()(
                &lOpts, &M, &N, &K,
                reinterpret_cast<scale_type<T>>(&alpha), descra,
                reinterpret_cast<cptr_type<T>>(values.get()),
                reinterpret_cast<const int*>(colIdx.get()),
                pB, pE,
                reinterpret_cast<cptr_type<T>>(rhs.get()), &ldb,
                reinterpret_cast<scale_type<T>>(&beta),
                reinterpret_cast<ptr_type<T>>(const_cast<T*>(out.get())), &ldc);
        }
    };

    getQueue().enqueue(func, out, lhs, rhs);

    return out;
}

#define INSTANTIATE_SPARSE(T)                                                           \
    template Array<T> matmul<T>(const common::SparseArray<T> lhs, const Array<T> rhs,   \
                                af_mat_prop optLhs, af_mat_prop optRhs);                \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}
