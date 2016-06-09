/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>

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
                                                const typename blas_base<T>::type,
                                                const T>::type;
#ifdef USE_MKL

// MKL
// sparse_status_t mkl_sparse_z_create_csr (
//                 sparse_matrix_t *A,
//                 sparse_index_base_t indexing,
//                 MKL_INT rows, MKL_INT cols,
//                 MKL_INT *rows_start, MKL_INT *rows_end,
//                 MKL_INT *col_indx,
//                 MKL_Complex16 *values);
//
// sparse_status_t mkl_sparse_z_mv (
//                 sparse_operation_t operation,
//                 MKL_Complex16 alpha,
//                 const sparse_matrix_t A,
//                 struct matrix_descr descr,
//                 const MKL_Complex16 *x,
//                 MKL_Complex16 beta,
//                 MKL_Complex16 *y);
//
// sparse_status_t mkl_sparse_z_mm (
//                 sparse_operation_t operation,
//                 MKL_Complex16 alpha,
//                 const sparse_matrix_t A,
//                 struct matrix_descr descr,
//                 sparse_layout_t layout,
//                 const MKL_Complex16 *x,
//                 MKL_INT columns, MKL_INT ldx,
//                 MKL_Complex16 beta,
//                 MKL_Complex16 *y,
//                 MKL_INT ldy);

template<typename T>
using create_csr_func_def = sparse_status_t (*)
                           (sparse_matrix_t *,
                            sparse_index_base_t,
                            int, int,
                            int *, int *, int*,
                            ptr_type<T>);

template<typename T>
using mv_func_def         = sparse_status_t (*)
                           (sparse_operation_t,
                            scale_type<T>,
                            const sparse_matrix_t,
                            struct matrix_descr,
                            cptr_type<T>,
                            scale_type<T>,
                            ptr_type<T>);

template<typename T>
using mm_func_def         = sparse_status_t (*)
                           (sparse_operation_t,
                            scale_type<T>,
                            const sparse_matrix_t,
                            struct matrix_descr,
                            sparse_layout_t,
                            cptr_type<T>,
                            int, int,
                            scale_type<T>,
                            ptr_type<T>, int);

#define SPARSE_FUNC_DEF( FUNC )                         \
template<typename T> FUNC##_func_def<T> FUNC##_func();

#define SPARSE_FUNC( FUNC, TYPE, PREFIX )               \
  template<> FUNC##_func_def<TYPE> FUNC##_func<TYPE>()  \
{ return &mkl_sparse_##PREFIX##_##FUNC; }

SPARSE_FUNC_DEF( create_csr )
SPARSE_FUNC(create_csr , float   , s)
SPARSE_FUNC(create_csr , double  , d)
SPARSE_FUNC(create_csr , cfloat  , c)
SPARSE_FUNC(create_csr , cdouble , z)

SPARSE_FUNC_DEF( mv )
SPARSE_FUNC(mv , float   , s)
SPARSE_FUNC(mv , double  , d)
SPARSE_FUNC(mv , cfloat  , c)
SPARSE_FUNC(mv , cdouble , z)

SPARSE_FUNC_DEF( mm )
SPARSE_FUNC(mm , float   , s)
SPARSE_FUNC(mm , double  , d)
SPARSE_FUNC(mm , cfloat  , c)
SPARSE_FUNC(mm , cdouble , z)

#else   // USE_MKL

// From mkl_spblas.h
typedef enum
{
    SPARSE_OPERATION_NON_TRANSPOSE          = 10,
    SPARSE_OPERATION_TRANSPOSE              = 11,
    SPARSE_OPERATION_CONJUGATE_TRANSPOSE    = 12,
} sparse_operation_t;

#endif  // USE_MKL

sparse_operation_t
toSparseTranspose(af_mat_prop opt)
{
    sparse_operation_t out = SPARSE_OPERATION_NON_TRANSPOSE;
    switch(opt) {
        case AF_MAT_NONE        : out = SPARSE_OPERATION_NON_TRANSPOSE;         break;
        case AF_MAT_TRANS       : out = SPARSE_OPERATION_TRANSPOSE;             break;
        case AF_MAT_CTRANS      : out = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;   break;
        default                 : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T, int value>
scale_type<T> getScale()
{
    static T val(value);
    //return (const typename blas_base<T>::type *)&val;
    return *(const scale_type<T>*)&val;
}

////////////////////////////////////////////////////////////////////////////////
#ifdef USE_MKL // Implementation using MKL
////////////////////////////////////////////////////////////////////////////////
template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    // MKL: CSRMM Does not support optRhs

    lhs.eval();
    rhs.eval();

    // Similar Operations to GEMM
    sparse_operation_t lOpts = toSparseTranspose(optLhs);

    int lRowDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;
    int lColDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;

    //Unsupported : (rOpts == SPARSE_OPERATION_NON_TRANSPOSE;) ? 1 : 0;
    static const int rColDim = 1;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[lRowDim];
    int N = rDims[rColDim];
    int K = lDims[lColDim];

    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));
    out.eval();

    auto func = [=] (Array<T> output, const SparseArray<T> left, const Array<T> right) {
        auto alpha = getScale<T, 1>();
        auto beta  = getScale<T, 0>();

        int ldb = right.strides()[1];
        int ldc = output.strides()[1];

        Array<T  > values = left.getValues();
        Array<int> rowIdx = left.getRowIdx();
        Array<int> colIdx = left.getColIdx();

        int *pB = rowIdx.get();
        int *pE = rowIdx.get() + 1;

        sparse_matrix_t csrLhs;
        create_csr_func<T>()(&csrLhs, SPARSE_INDEX_BASE_ZERO, M, K,
                             pB, pE, colIdx.get(),
                             reinterpret_cast<ptr_type<T>>(values.get()));

        struct matrix_descr descrLhs;
        descrLhs.type = SPARSE_MATRIX_TYPE_GENERAL;

        mkl_sparse_optimize(csrLhs);

        if(rDims[rColDim] == 1) {
            mkl_sparse_set_mv_hint(csrLhs, lOpts, descrLhs, 1);
            mv_func<T>()(
                lOpts, alpha,
                csrLhs, descrLhs,
                reinterpret_cast<cptr_type<T>>(right.get()),
                beta,
                reinterpret_cast<ptr_type<T>>(output.get()));
        } else {
            mkl_sparse_set_mm_hint(csrLhs, lOpts, descrLhs, SPARSE_LAYOUT_COLUMN_MAJOR, N, 1);
            mm_func<T>()(
                lOpts, alpha,
                csrLhs, descrLhs, SPARSE_LAYOUT_COLUMN_MAJOR,
                reinterpret_cast<cptr_type<T>>(right.get()),
                N, ldb, beta,
                reinterpret_cast<ptr_type<T>>(output.get()), ldc);
        }
        mkl_sparse_destroy(csrLhs);
    };

    getQueue().enqueue(func, out, lhs, rhs);

    return out;
}

////////////////////////////////////////////////////////////////////////////////
#else // Implementation without using MKL
////////////////////////////////////////////////////////////////////////////////

template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    // TODO: Make a CPU Implementation for this
    // Make separate function for MV and MM
    // No need to support optRhs
    lhs.eval();
    rhs.eval();

    // Similar Operations to GEMM
    sparse_operation_t lOpts = toSparseTranspose(optLhs);

    int lRowDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;
    // Commenting to avoid unused variable warnings
    //int lColDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;

    //Unsupported : (rOpts == SPARSE_OPERATION_NON_TRANSPOSE;) ? 1 : 0;
    static const int rColDim = 1;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[lRowDim];
    int N = rDims[rColDim];
    // Commenting to avoid unused variable warnings
    //int K = lDims[lColDim];

    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));
    out.eval();

    // Commenting to avoid unused variable warnings
    //auto func = [=] (Array<T> output, const SparseArray<T> left, const Array<T> right) {
    //    auto alpha = getScale<T, 1>();
    //    auto beta  = getScale<T, 0>();

    //    int ldb = right.strides()[1];
    //    int ldc = output.strides()[1];

    //    Array<T  > values = left.getValues();
    //    Array<int> rowIdx = left.getRowIdx();
    //    Array<int> colIdx = left.getColIdx();

    //    if(rDims[rColDim] == 1) {
    //        // Call MV
    //    } else {
    //        // Call MM
    //    }
    //};

    //getQueue().enqueue(func, out, lhs, rhs);

    return out;
}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_SPARSE(T)                                                           \
    template Array<T> matmul<T>(const common::SparseArray<T> lhs, const Array<T> rhs,   \
                                af_mat_prop optLhs, af_mat_prop optRhs);                \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}
