/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>

#ifdef USE_MKL
#include <mkl_spblas.h>
#endif

#include <common/complex.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <cassert>
#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cpu {

#ifdef USE_MKL
using sp_cfloat  = MKL_Complex8;
using sp_cdouble = MKL_Complex16;
#else
using sp_cfloat  = cfloat;
using sp_cdouble = cdouble;

// From mkl_spblas.h
typedef enum {
    SPARSE_OPERATION_NON_TRANSPOSE       = 10,
    SPARSE_OPERATION_TRANSPOSE           = 11,
    SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12,
} sparse_operation_t;
#endif

template<typename T, class Enable = void>
struct blas_base {
    using type = T;
};

template<typename T>
struct blas_base<T,
                 typename std::enable_if<common::is_complex<T>::value>::type> {
    using type = typename std::conditional<std::is_same<T, cdouble>::value,
                                           sp_cdouble, sp_cfloat>::type;
};

template<typename T>
using cptr_type = typename std::conditional<common::is_complex<T>::value,
                                            const typename blas_base<T>::type *,
                                            const T *>::type;
template<typename T>
using ptr_type =
    typename std::conditional<common::is_complex<T>::value,
                              typename blas_base<T>::type *, T *>::type;
template<typename T>
using scale_type =
    typename std::conditional<common::is_complex<T>::value,
                              const typename blas_base<T>::type, const T>::type;

template<typename To, typename Ti>
auto getScaleValue(Ti val) -> std::remove_cv_t<To> {
    return static_cast<std::remove_cv_t<To>>(val);
}

template<typename T, int value>
scale_type<T> getScale() {  // NOLINT(readability-const-return-type)
    static T val(value);
    return getScaleValue<scale_type<T>, T>(val);
}

sparse_operation_t toSparseTranspose(af_mat_prop opt) {
    sparse_operation_t out = SPARSE_OPERATION_NON_TRANSPOSE;
    switch (opt) {
        case AF_MAT_NONE: out = SPARSE_OPERATION_NON_TRANSPOSE; break;
        case AF_MAT_TRANS: out = SPARSE_OPERATION_TRANSPOSE; break;
        case AF_MAT_CTRANS: out = SPARSE_OPERATION_CONJUGATE_TRANSPOSE; break;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

#ifdef USE_MKL

template<>
sp_cfloat getScaleValue<const sp_cfloat, cfloat>(cfloat val) {
    sp_cfloat ret;
    ret.real = val.real();
    ret.imag = val.imag();
    return ret;
}

template<>
sp_cdouble getScaleValue<const sp_cdouble, cdouble>(cdouble val) {
    sp_cdouble ret;
    ret.real = val.real();
    ret.imag = val.imag();
    return ret;
}

// sparse_status_t mkl_sparse_z_create_csr (
//                 sparse_matrix_t *A,
//                 sparse_index_base_t indexing,
//                 MKL_INT rows, MKL_INT cols,
//                 MKL_INT *rows_start, MKL_INT *rows_end,
//                 MKL_INT *col_indx,
//                 MKL_Complex16 *values);

template<typename T>
using create_csr_func_def = sparse_status_t (*)(sparse_matrix_t *,
                                                sparse_index_base_t, int, int,
                                                int *, int *, int *,
                                                ptr_type<T>);

#define SPARSE_FUNC_DEF(FUNC) \
    template<typename T>      \
    FUNC##_func_def<T> FUNC##_func();

SPARSE_FUNC_DEF(create_csr)

#undef SPARSE_FUNC_DEF

#define SPARSE_FUNC(FUNC, TYPE, PREFIX)         \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &mkl_sparse_##PREFIX##_##FUNC;   \
    }

SPARSE_FUNC(create_csr, float, s)
SPARSE_FUNC(create_csr, double, d)
SPARSE_FUNC(create_csr, cfloat, c)
SPARSE_FUNC(create_csr, cdouble, z)

#undef SPARSE_FUNC

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
using mv_func_def = sparse_status_t (*)(const sparse_operation_t, scale_type<T>,
                                        const sparse_matrix_t, matrix_descr,
                                        cptr_type<T>, scale_type<T>,
                                        ptr_type<T>);

template<typename T>
using mm_func_def = sparse_status_t (*)(const sparse_operation_t, scale_type<T>,
                                        const sparse_matrix_t, matrix_descr,
                                        sparse_layout_t, cptr_type<T>, int, int,
                                        scale_type<T>, ptr_type<T>, int);

#define SPARSE_FUNC_DEF(FUNC) \
    template<typename T>      \
    FUNC##_func_def<T> FUNC##_func();

#define SPARSE_FUNC(FUNC, TYPE, PREFIX)         \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &mkl_sparse_##PREFIX##_##FUNC;   \
    }

SPARSE_FUNC_DEF(mv)
SPARSE_FUNC(mv, float, s)
SPARSE_FUNC(mv, double, d)
SPARSE_FUNC(mv, cfloat, c)
SPARSE_FUNC(mv, cdouble, z)

SPARSE_FUNC_DEF(mm)
SPARSE_FUNC(mm, float, s)
SPARSE_FUNC(mm, double, d)
SPARSE_FUNC(mm, cfloat, c)
SPARSE_FUNC(mm, cdouble, z)

template<typename T>
Array<T> matmul(const common::SparseArray<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs) {
    // MKL: CSRMM Does not support optRhs
    UNUSED(optRhs);

    // Similar Operations to GEMM
    sparse_operation_t lOpts = toSparseTranspose(optLhs);

    int lRowDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;
    // int lColDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;

    // Unsupported : (rOpts == SPARSE_OPERATION_NON_TRANSPOSE;) ? 1 : 0;
    static const int rColDim = 1;

    const dim4 &lDims = lhs.dims();
    const dim4 &rDims = rhs.dims();

    int M = lDims[lRowDim];
    int N = rDims[rColDim];
    // int K = lDims[lColDim];

    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));

    auto func = [=](Param<T> output, CParam<T> values, CParam<int> rowIdx,
                    CParam<int> colIdx, const dim_t sdim0, const dim_t sdim1,
                    CParam<T> right) {
        auto alpha = getScale<T, 1>();
        auto beta  = getScale<T, 0>();

        int ldb = right.strides(1);
        int ldc = output.strides(1);

        int *pB = const_cast<int *>(rowIdx.get());
        int *pE = pB + 1;
        T *vptr = const_cast<T *>(values.get());

        sparse_matrix_t csrLhs;
        create_csr_func<T>()(&csrLhs, SPARSE_INDEX_BASE_ZERO, sdim0, sdim1, pB,
                             pE, const_cast<int *>(colIdx.get()),
                             reinterpret_cast<ptr_type<T>>(vptr));

        struct matrix_descr descrLhs {};
        descrLhs.type = SPARSE_MATRIX_TYPE_GENERAL;

        mkl_sparse_optimize(csrLhs);

        if (rDims[rColDim] == 1) {
            mkl_sparse_set_mv_hint(csrLhs, lOpts, descrLhs, 1);
            mv_func<T>()(lOpts, alpha, csrLhs, descrLhs,
                         reinterpret_cast<cptr_type<T>>(right.get()), beta,
                         reinterpret_cast<ptr_type<T>>(output.get()));
        } else {
            mkl_sparse_set_mm_hint(csrLhs, lOpts, descrLhs,
                                   SPARSE_LAYOUT_COLUMN_MAJOR, N, 1);
            mm_func<T>()(
                lOpts, alpha, csrLhs, descrLhs, SPARSE_LAYOUT_COLUMN_MAJOR,
                reinterpret_cast<cptr_type<T>>(right.get()), N, ldb, beta,
                reinterpret_cast<ptr_type<T>>(output.get()), ldc);
        }
        mkl_sparse_destroy(csrLhs);
    };

    const Array<T> values   = lhs.getValues();
    const Array<int> rowIdx = lhs.getRowIdx();
    const Array<int> colIdx = lhs.getColIdx();
    af::dim4 ldims          = lhs.dims();

    getQueue().enqueue(func, out, values, rowIdx, colIdx, ldims[0], ldims[1],
                       rhs);

    return out;
}

#else  // #if USE_MKL

template<typename T>
T getConjugate(const T &in) {
    // For non-complex types return same
    return in;
}

template<>
cfloat getConjugate(const cfloat &in) {
    return std::conj(in);
}

template<>
cdouble getConjugate(const cdouble &in) {
    return std::conj(in);
}

template<typename T, bool conjugate>
void mv(Param<T> output, CParam<T> values, CParam<int> rowIdx,
        CParam<int> colIdx, CParam<T> right, int M) {
    const T *valPtr   = values.get();
    const int *rowPtr = rowIdx.get();
    const int *colPtr = colIdx.get();
    const T *rightPtr = right.get();

    T *outPtr = output.get();

    // Output Array Created is a zero value Array
    // Hence, no need to initialize to zero here
    for (int i = 0; i < M; ++i) {
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            // If stride[0] of right is not 1 then rightPtr[colPtr[j]*stride]
            if (conjugate) {
                outPtr[i] += getConjugate(valPtr[j]) * rightPtr[colPtr[j]];
            } else {
                outPtr[i] += valPtr[j] * rightPtr[colPtr[j]];
            }
        }
    }
}

template<typename T, bool conjugate>
void mtv(Param<T> output, CParam<T> values, CParam<int> rowIdx,
         CParam<int> colIdx, CParam<T> right, int M) {
    UNUSED(M);

    const T *valPtr   = values.get();
    const int *rowPtr = rowIdx.get();
    const int *colPtr = colIdx.get();
    const T *rightPtr = right.get();
    T *outPtr         = output.get();

    // Output Array Created is a zero value Array
    // Hence, no need to initialize to zero here
    for (int i = 0; i < rowIdx.dims(0) - 1; ++i) {
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            // If stride[0] of right is not 1 then rightPtr[i*stride]
            if (conjugate) {
                outPtr[colPtr[j]] += getConjugate(valPtr[j]) * rightPtr[i];
            } else {
                outPtr[colPtr[j]] += valPtr[j] * rightPtr[i];
            }
        }
    }
}

template<typename T, bool conjugate>
void mm(Param<T> output, CParam<T> values, CParam<int> rowIdx,
        CParam<int> colIdx, CParam<T> right, int M, int N, int ldb, int ldc) {
    UNUSED(M);
    const T *valPtr   = values.get();
    const int *rowPtr = rowIdx.get();
    const int *colPtr = colIdx.get();
    const T *rightPtr = right.get();
    T *outPtr         = output.get();

    for (int o = 0; o < N; ++o) {
        for (int i = 0; i < rowIdx.dims(0) - 1; ++i) {
            outPtr[i] = scalar<T>(0);
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                // If stride[0] of right is not 1 then
                // rightPtr[colPtr[j]*stride]
                if (conjugate) {
                    outPtr[i] += getConjugate(valPtr[j]) * rightPtr[colPtr[j]];
                } else {
                    outPtr[i] += valPtr[j] * rightPtr[colPtr[j]];
                }
            }
        }
        rightPtr += ldb;
        outPtr += ldc;
    }
}

template<typename T, bool conjugate>
void mtm(Param<T> output, CParam<T> values, CParam<int> rowIdx,
         CParam<int> colIdx, CParam<T> right, int M, int N, int ldb, int ldc) {
    const T *valPtr   = values.get();
    const int *rowPtr = rowIdx.get();
    const int *colPtr = colIdx.get();
    const T *rightPtr = right.get();
    T *outPtr         = output.get();

    for (int o = 0; o < N; ++o) {
        for (int i = 0; i < M; ++i) { outPtr[i] = scalar<T>(0); }

        for (int i = 0; i < rowIdx.dims(0) - 1; ++i) {
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                // If stride[0] of right is not 1 then rightPtr[i*stride]
                if (conjugate) {
                    outPtr[colPtr[j]] += getConjugate(valPtr[j]) * rightPtr[i];
                } else {
                    outPtr[colPtr[j]] += valPtr[j] * rightPtr[i];
                }
            }
        }
        rightPtr += ldb;
        outPtr += ldc;
    }
}

template<typename T>
Array<T> matmul(const common::SparseArray<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs) {
    UNUSED(optRhs);

    // Similar Operations to GEMM
    sparse_operation_t lOpts = toSparseTranspose(optLhs);

    int lRowDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;

    static const int rColDim = 1;

    const dim4 &lDims = lhs.dims();
    const dim4 &rDims = rhs.dims();
    int M             = lDims[lRowDim];
    int N             = rDims[rColDim];

    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));

    auto func = [=](Param<T> output, CParam<T> values, CParam<int> rowIdx,
                    CParam<int> colIdx, CParam<T> right) {
        int ldb = right.strides(1);
        int ldc = output.strides(1);

        if (rDims[rColDim] == 1) {
            if (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) {
                mv<T, false>(output, values, rowIdx, colIdx, right, M);
            } else if (lOpts == SPARSE_OPERATION_TRANSPOSE) {
                mtv<T, false>(output, values, rowIdx, colIdx, right, M);
            } else if (lOpts == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mtv<T, true>(output, values, rowIdx, colIdx, right, M);
            }
        } else {
            if (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) {
                mm<T, false>(output, values, rowIdx, colIdx, right, M, N, ldb,
                             ldc);
            } else if (lOpts == SPARSE_OPERATION_TRANSPOSE) {
                mtm<T, false>(output, values, rowIdx, colIdx, right, M, N, ldb,
                              ldc);
            } else if (lOpts == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mtm<T, true>(output, values, rowIdx, colIdx, right, M, N, ldb,
                             ldc);
            }
        }
    };

    const Array<T> values   = lhs.getValues();
    const Array<int> rowIdx = lhs.getRowIdx();
    const Array<int> colIdx = lhs.getColIdx();

    getQueue().enqueue(func, out, values, rowIdx, colIdx, rhs);

    return out;
}

#endif  // #if USE_MKL

#define INSTANTIATE_SPARSE(T)                                            \
    template Array<T> matmul<T>(const common::SparseArray<T> &lhs,       \
                                const Array<T> &rhs, af_mat_prop optLhs, \
                                af_mat_prop optRhs);

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}  // namespace cpu
}  // namespace arrayfire
