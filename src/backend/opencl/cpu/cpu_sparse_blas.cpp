/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_LINEAR_ALGEBRA)
#include <cpu/cpu_sparse_blas.hpp>

#include <common/complex.hpp>
#include <complex.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>

#include <stdexcept>
#include <string>

using arrayfire::common::is_complex;

using std::add_const;
using std::add_pointer;
using std::conditional;
using std::enable_if;
using std::is_floating_point;
using std::is_same;
using std::remove_const;

namespace arrayfire {
namespace opencl {
namespace cpu {

template<typename T, class Enable = void>
struct blas_base {
    using type = T;
};

template<typename T>
struct blas_base<T, typename enable_if<is_complex<T>::value>::type> {
    using type = typename conditional<is_same<T, cdouble>::value, sp_cdouble,
                                      sp_cfloat>::type;
};

template<typename T>
using cptr_type =
    typename conditional<is_complex<T>::value,
                         const typename blas_base<T>::type *, const T *>::type;
template<typename T>
using ptr_type = typename conditional<is_complex<T>::value,
                                      typename blas_base<T>::type *, T *>::type;
template<typename T>
using scale_type =
    typename conditional<is_complex<T>::value,
                         const typename blas_base<T>::type, const T>::type;

template<typename To, typename Ti>
auto getScaleValue(Ti val) -> std::remove_cv_t<To> {
    return static_cast<std::remove_cv_t<To>>(val);
}

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
using create_csr_func_def = sparse_status_t (*)(sparse_matrix_t *,
                                                sparse_index_base_t, int, int,
                                                int *, int *, int *,
                                                ptr_type<T>);

template<typename T>
using mv_func_def = sparse_status_t (*)(sparse_operation_t, scale_type<T>,
                                        const sparse_matrix_t, matrix_descr,
                                        cptr_type<T>, scale_type<T>,
                                        ptr_type<T>);

template<typename T>
using mm_func_def = sparse_status_t (*)(sparse_operation_t, scale_type<T>,
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

SPARSE_FUNC_DEF(create_csr)
SPARSE_FUNC(create_csr, float, s)
SPARSE_FUNC(create_csr, double, d)
SPARSE_FUNC(create_csr, cfloat, c)
SPARSE_FUNC(create_csr, cdouble, z)

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

#undef SPARSE_FUNC
#undef SPARSE_FUNC_DEF

template<>
sp_cfloat getScaleValue<const sp_cfloat, cfloat>(cfloat val) {
    sp_cfloat ret;
    ret.real = val.s[0];
    ret.imag = val.s[1];
    return ret;
}

template<>
sp_cdouble getScaleValue<const sp_cdouble, cdouble>(cdouble val) {
    sp_cdouble ret;
    ret.real = val.s[0];
    ret.imag = val.s[1];
    return ret;
}

#else  // USE_MKL

// From mkl_spblas.h
typedef enum {
    SPARSE_OPERATION_NON_TRANSPOSE       = 10,
    SPARSE_OPERATION_TRANSPOSE           = 11,
    SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12,
} sparse_operation_t;

#endif  // USE_MKL

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

template<typename T, int value>
scale_type<T> getScale() {  // NOLINT(readability-const-return-type)
    thread_local T val = scalar<T>(value);
    return getScaleValue<scale_type<T>, T>(val);
}

////////////////////////////////////////////////////////////////////////////////
#ifdef USE_MKL  // Implementation using MKL
////////////////////////////////////////////////////////////////////////////////
template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs) {
    // MKL: CSRMM Does not support optRhs
    UNUSED(optRhs);

    lhs.eval();
    rhs.eval();

    // Similar Operations to GEMM
    sparse_operation_t lOpts = toSparseTranspose(optLhs);

    int lRowDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;
    // int lColDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;

    // Unsupported : (rOpts == SPARSE_OPERATION_NON_TRANSPOSE;) ? 1 : 0;
    static const int rColDim = 1;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = lDims[lRowDim];
    int N      = rDims[rColDim];
    // int K = lDims[lColDim];

    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));
    out.eval();

    auto alpha = getScale<T, 1>();
    auto beta  = getScale<T, 0>();

    int ldb = rhs.strides()[1];
    int ldc = out.strides()[1];

    // get host pointers from mapped memory
    mapped_ptr<T> rhsPtr = rhs.getMappedPtr(CL_MAP_READ);
    mapped_ptr<T> outPtr = out.getMappedPtr();

    Array<T> values   = lhs.getValues();
    Array<int> rowIdx = lhs.getRowIdx();
    Array<int> colIdx = lhs.getColIdx();

    mapped_ptr<T> vPtr   = values.getMappedPtr();
    mapped_ptr<int> rPtr = rowIdx.getMappedPtr();
    mapped_ptr<int> cPtr = colIdx.getMappedPtr();
    int *pB              = rPtr.get();
    int *pE              = rPtr.get() + 1;

    sparse_matrix_t csrLhs;
    create_csr_func<T>()(&csrLhs, SPARSE_INDEX_BASE_ZERO, lhs.dims()[0],
                         lhs.dims()[1], pB, pE, cPtr.get(),
                         reinterpret_cast<ptr_type<T>>(vPtr.get()));

    struct matrix_descr descrLhs {};
    descrLhs.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_optimize(csrLhs);

    if (rDims[rColDim] == 1) {
        mkl_sparse_set_mv_hint(csrLhs, lOpts, descrLhs, 1);
        mv_func<T>()(lOpts, alpha, csrLhs, descrLhs,
                     reinterpret_cast<cptr_type<T>>(rhsPtr.get()), beta,
                     reinterpret_cast<ptr_type<T>>(outPtr.get()));
    } else {
        mkl_sparse_set_mm_hint(csrLhs, lOpts, descrLhs,
                               SPARSE_LAYOUT_COLUMN_MAJOR, N, 1);
        mm_func<T>()(lOpts, alpha, csrLhs, descrLhs, SPARSE_LAYOUT_COLUMN_MAJOR,
                     reinterpret_cast<cptr_type<T>>(rhsPtr.get()), N, ldb, beta,
                     reinterpret_cast<ptr_type<T>>(outPtr.get()), ldc);
    }
    mkl_sparse_destroy(csrLhs);

    return out;
}

////////////////////////////////////////////////////////////////////////////////
#else  // Implementation without using MKL
////////////////////////////////////////////////////////////////////////////////

template<typename T>
T getConjugate(const T &in) {
    // For non-complex types return same
    return in;
}

template<>
cfloat getConjugate(const cfloat &in) {
    cfloat val;
    val.s[0] = in.s[0];
    val.s[1] = -in.s[1];
    return val;
}

template<>
cdouble getConjugate(const cdouble &in) {
    cdouble val;
    val.s[0] = in.s[0];
    val.s[1] = -in.s[1];
    return val;
}

template<typename T, bool conjugate>
void mv(Array<T> output, const Array<T> values, const Array<int> rowIdx,
        const Array<int> colIdx, const Array<T> right, int M) {
    UNUSED(M);
    mapped_ptr<T> oPtr   = output.getMappedPtr();
    mapped_ptr<T> rhtPtr = right.getMappedPtr();
    mapped_ptr<T> vPtr   = values.getMappedPtr();
    mapped_ptr<int> rPtr = rowIdx.getMappedPtr();
    mapped_ptr<int> cPtr = colIdx.getMappedPtr();

    T const *const valPtr   = vPtr.get();
    int const *const rowPtr = rPtr.get();
    int const *const colPtr = cPtr.get();
    T const *const rhsPtr   = rhtPtr.get();
    T *const outPtr         = oPtr.get();

    for (int i = 0; i < rowIdx.dims()[0] - 1; ++i) {
        outPtr[i] = scalar<T>(0);
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            // If stride[0] of right is not 1 then rhsPtr[colPtr[j]*stride]
            if (conjugate) {
                outPtr[i] =
                    outPtr[i] + getConjugate(valPtr[j]) * rhsPtr[colPtr[j]];
            } else {
                outPtr[i] = outPtr[i] + valPtr[j] * rhsPtr[colPtr[j]];
            }
        }
    }
}

template<typename T, bool conjugate>
void mtv(Array<T> output, const Array<T> values, const Array<int> rowIdx,
         const Array<int> colIdx, const Array<T> right, int M) {
    mapped_ptr<T> oPtr   = output.getMappedPtr();
    mapped_ptr<T> rhtPtr = right.getMappedPtr();
    mapped_ptr<T> vPtr   = values.getMappedPtr();
    mapped_ptr<int> rPtr = rowIdx.getMappedPtr();
    mapped_ptr<int> cPtr = colIdx.getMappedPtr();

    T const *const valPtr   = vPtr.get();
    int const *const rowPtr = rPtr.get();
    int const *const colPtr = cPtr.get();
    T const *const rhsPtr   = rhtPtr.get();
    T *const outPtr         = oPtr.get();

    for (int i = 0; i < M; ++i) { outPtr[i] = scalar<T>(0); }

    for (int i = 0; i < rowIdx.dims()[0] - 1; ++i) {
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            // If stride[0] of right is not 1 then rhsPtr[i*stride]
            if (conjugate) {
                outPtr[colPtr[j]] =
                    outPtr[colPtr[j]] + getConjugate(valPtr[j]) * rhsPtr[i];
            } else {
                outPtr[colPtr[j]] = outPtr[colPtr[j]] + valPtr[j] * rhsPtr[i];
            }
        }
    }
}

template<typename T, bool conjugate>
void mm(Array<T> output, const Array<T> values, const Array<int> rowIdx,
        const Array<int> colIdx, const Array<T> right, int M, int N, int ldb,
        int ldc) {
    UNUSED(M);
    mapped_ptr<T> oPtr   = output.getMappedPtr();
    mapped_ptr<T> rhtPtr = right.getMappedPtr();
    mapped_ptr<T> vPtr   = values.getMappedPtr();
    mapped_ptr<int> rPtr = rowIdx.getMappedPtr();
    mapped_ptr<int> cPtr = colIdx.getMappedPtr();

    T const *const valPtr   = vPtr.get();
    int const *const rowPtr = rPtr.get();
    int const *const colPtr = cPtr.get();
    T const *rhsPtr         = rhtPtr.get();
    T *outPtr               = oPtr.get();

    for (int o = 0; o < N; ++o) {
        for (int i = 0; i < rowIdx.dims()[0] - 1; ++i) {
            outPtr[i] = scalar<T>(0);
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                // If stride[0] of right is not 1 then rhsPtr[colPtr[j]*stride]
                if (conjugate) {
                    outPtr[i] =
                        outPtr[i] + getConjugate(valPtr[j]) * rhsPtr[colPtr[j]];
                } else {
                    outPtr[i] = outPtr[i] + valPtr[j] * rhsPtr[colPtr[j]];
                }
            }
        }
        rhsPtr += ldb;
        outPtr += ldc;
    }
}

template<typename T, bool conjugate>
void mtm(Array<T> output, const Array<T> values, const Array<int> rowIdx,
         const Array<int> colIdx, const Array<T> right, int M, int N, int ldb,
         int ldc) {
    mapped_ptr<T> oPtr   = output.getMappedPtr();
    mapped_ptr<T> rhtPtr = right.getMappedPtr();
    mapped_ptr<T> vPtr   = values.getMappedPtr();
    mapped_ptr<int> rPtr = rowIdx.getMappedPtr();
    mapped_ptr<int> cPtr = colIdx.getMappedPtr();

    T const *const valPtr   = vPtr.get();
    int const *const rowPtr = rPtr.get();
    int const *const colPtr = cPtr.get();
    T const *rhsPtr         = rhtPtr.get();
    T *outPtr               = oPtr.get();

    for (int o = 0; o < N; ++o) {
        for (int i = 0; i < M; ++i) { outPtr[i] = scalar<T>(0); }

        for (int i = 0; i < rowIdx.dims()[0] - 1; ++i) {
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                // If stride[0] of right is not 1 then rhsPtr[i*stride]
                if (conjugate) {
                    outPtr[colPtr[j]] =
                        outPtr[colPtr[j]] + getConjugate(valPtr[j]) * rhsPtr[i];
                } else {
                    outPtr[colPtr[j]] =
                        outPtr[colPtr[j]] + valPtr[j] * rhsPtr[i];
                }
            }
        }
        rhsPtr += ldb;
        outPtr += ldc;
    }
}
template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs) {
    UNUSED(optRhs);
    lhs.eval();
    rhs.eval();

    // Similar Operations to GEMM
    sparse_operation_t lOpts = toSparseTranspose(optLhs);

    int lRowDim = (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;

    static const int rColDim = 1;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = lDims[lRowDim];
    int N      = rDims[rColDim];

    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));
    out.eval();

    int ldb = rhs.strides()[1];
    int ldc = out.strides()[1];

    Array<T> values   = lhs.getValues();
    Array<int> rowIdx = lhs.getRowIdx();
    Array<int> colIdx = lhs.getColIdx();

    if (rDims[rColDim] == 1) {
        if (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) {
            mv<T, false>(out, values, rowIdx, colIdx, rhs, M);
        } else if (lOpts == SPARSE_OPERATION_TRANSPOSE) {
            mtv<T, false>(out, values, rowIdx, colIdx, rhs, M);
        } else if (lOpts == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
            mtv<T, true>(out, values, rowIdx, colIdx, rhs, M);
        }
    } else {
        if (lOpts == SPARSE_OPERATION_NON_TRANSPOSE) {
            mm<T, false>(out, values, rowIdx, colIdx, rhs, M, N, ldb, ldc);
        } else if (lOpts == SPARSE_OPERATION_TRANSPOSE) {
            mtm<T, false>(out, values, rowIdx, colIdx, rhs, M, N, ldb, ldc);
        } else if (lOpts == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
            mtm<T, true>(out, values, rowIdx, colIdx, rhs, M, N, ldb, ldc);
        }
    }

    return out;
}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_SPARSE(T)                                           \
    template Array<T> matmul<T>(const common::SparseArray<T> lhs,       \
                                const Array<T> rhs, af_mat_prop optLhs, \
                                af_mat_prop optRhs);

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_SPARSE

}  // namespace cpu
}  // namespace opencl
}  // namespace arrayfire
#endif  // WITH_LINEAR_ALGEBRA
