/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <sparse_common.hpp>
#include <testHelpers.hpp>

using af::allTrue;
using af::array;
using af::deviceMemInfo;
using af::dim4;
using af::dtype_traits;
using af::identity;
using af::randu;
using af::span;
using af::seq;

#define SPARSE_TESTS(T, eps)                                                \
    TEST(Sparse, T##Square) { sparseTester<T>(1000, 1000, 100, 5, eps); }   \
    TEST(Sparse, T##RectMultiple) {                                         \
        sparseTester<T>(2048, 1024, 512, 3, eps);                           \
    }                                                                       \
    TEST(Sparse, T##RectDense) { sparseTester<T>(500, 1000, 250, 1, eps); } \
    TEST(Sparse, T##MatVec) { sparseTester<T>(625, 1331, 1, 2, eps); }      \
    TEST(Sparse, Transpose_##T##MatVec) {                                   \
        sparseTransposeTester<T>(625, 1331, 1, 2, eps);                     \
    }                                                                       \
    TEST(Sparse, Transpose_##T##Square) {                                   \
        sparseTransposeTester<T>(1000, 1000, 100, 5, eps);                  \
    }                                                                       \
    TEST(Sparse, Transpose_##T##RectMultiple) {                             \
        sparseTransposeTester<T>(2048, 1024, 512, 3, eps);                  \
    }                                                                       \
    TEST(Sparse, Transpose_##T##RectDense) {                                \
        sparseTransposeTester<T>(453, 751, 397, 1, eps);                    \
    }                                                                       \
    TEST(Sparse, T##ConvertCSR) { convertCSR<T>(2345, 5678, 0.5); }

SPARSE_TESTS(float, 1E-3)
SPARSE_TESTS(double, 1E-5)
SPARSE_TESTS(cfloat, 1E-3)
SPARSE_TESTS(cdouble, 1E-5)

#undef SPARSE_TESTS

#define CREATE_TESTS(STYPE) \
    TEST(Sparse, Create_##STYPE) { createFunction<STYPE>(); }

CREATE_TESTS(AF_STORAGE_CSR)
CREATE_TESTS(AF_STORAGE_COO)

#undef CREATE_TESTS

TEST(Sparse, Create_AF_STORAGE_CSC) {
    array d = identity(3, 3);

    af_array out = 0;
    ASSERT_EQ(AF_ERR_ARG,
              af_create_sparse_array_from_dense(&out, d.get(), AF_STORAGE_CSC));

    if (out != 0) af_release_array(out);
}

#define CAST_TESTS_TYPES(Ti, To, SUFFIX, M, N, F) \
    TEST(Sparse, Cast_##Ti##_##To##_##SUFFIX) {   \
        sparseCastTester<Ti, To>(M, N, F);        \
    }

#define CAST_TESTS(Ti, To)                     \
    CAST_TESTS_TYPES(Ti, To, 1, 1000, 1000, 5) \
    CAST_TESTS_TYPES(Ti, To, 2, 512, 1024, 2)

CAST_TESTS(float, float)
CAST_TESTS(float, double)
CAST_TESTS(float, cfloat)
CAST_TESTS(float, cdouble)

CAST_TESTS(double, float)
CAST_TESTS(double, double)
CAST_TESTS(double, cfloat)
CAST_TESTS(double, cdouble)

CAST_TESTS(cfloat, cfloat)
CAST_TESTS(cfloat, cdouble)

CAST_TESTS(cdouble, cfloat)
CAST_TESTS(cdouble, cdouble)

TEST(Sparse, ISSUE_1745) {
    using af::where;

    array A    = randu(4, 4);
    A(1, span) = 0;
    A(2, span) = 0;

    array idx     = where(A);
    array data    = A(idx);
    array row_idx = (idx / A.dims()[0]).as(s64);
    array col_idx = (idx % A.dims()[0]).as(s64);

    af_array A_sparse;
    ASSERT_EQ(AF_ERR_ARG, af_create_sparse_array(
                              &A_sparse, A.dims(0), A.dims(1), data.get(),
                              row_idx.get(), col_idx.get(), AF_STORAGE_CSR));
}

TEST(Sparse, ISSUE_1918) {
    array reference(2,2);
    reference(0, span) = 0;
    reference(1, span) = 2;
    array output;
    float value[] = { 1, 1, 2, 2 };
    int index[] = { -1, 1, 2 };
    int row[] = { 0, 2, 2, 0, 0, 2 };
    int col[] = { 0, 1, 0, 1 };
    array values(4, 1, value, afHost);
    array rows(6, 1, row, afHost);
    array cols(4, 1, col, afHost);
    array S;
  
    S = sparse(2, 2, values(seq(2, 3)), rows(seq(3, 5)), cols(seq(2, 3)));
    output = dense(S);

    ASSERT_ARRAYS_EQ(reference, output);
}

TEST(Sparse, ISSUE_2134_COO) {
    int rows[]     = {0, 0, 0, 1, 1, 2, 2};
    int cols[]     = {0, 1, 2, 0, 1, 0, 2};
    float values[] = {3, 3, 4, 3, 10, 4, 3};
    array row(7, rows);
    array col(7, cols);
    array value(7, values);
    af_array A = 0;
    EXPECT_EQ(AF_ERR_SIZE,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_CSR));
    if (A != 0) af_release_array(A);
    A = 0;
    EXPECT_EQ(AF_ERR_SIZE,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_CSC));
    if (A != 0) af_release_array(A);
    A = 0;
    EXPECT_EQ(AF_SUCCESS,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_COO));
    if (A != 0) af_release_array(A);
}

TEST(Sparse, ISSUE_2134_CSR) {
    int rows[]     = {0, 3, 5, 7};
    int cols[]     = {0, 1, 2, 0, 1, 0, 2};
    float values[] = {3, 3, 4, 3, 10, 4, 3};
    array row(4, rows);
    array col(7, cols);
    array value(7, values);
    af_array A = 0;
    EXPECT_EQ(AF_SUCCESS,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_CSR));
    if (A != 0) af_release_array(A);
    A = 0;
    EXPECT_EQ(AF_ERR_SIZE,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_CSC));
    if (A != 0) af_release_array(A);
    A = 0;
    EXPECT_EQ(AF_ERR_SIZE,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_COO));
    if (A != 0) af_release_array(A);
}

TEST(Sparse, ISSUE_2134_CSC) {
    int rows[]     = {0, 0, 0, 1, 1, 2, 2};
    int cols[]     = {0, 3, 5, 7};
    float values[] = {3, 3, 4, 3, 10, 4, 3};
    array row(7, rows);
    array col(4, cols);
    array value(7, values);
    af_array A = 0;
    EXPECT_EQ(AF_ERR_SIZE,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_CSR));
    if (A != 0) af_release_array(A);
    A = 0;
    EXPECT_EQ(AF_SUCCESS,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_CSC));
    if (A != 0) af_release_array(A);
    A = 0;
    EXPECT_EQ(AF_ERR_SIZE,
              af_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), AF_STORAGE_COO));
    if (A != 0) af_release_array(A);
}

template<typename T>
class Sparse : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> SparseTypes;
TYPED_TEST_SUITE(Sparse, SparseTypes);

TYPED_TEST(Sparse, DeepCopy) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    cleanSlate();

    array s;
    {
        // Create a sparse array from a dense array. Make sure that the dense
        // arrays are removed
        array dense = randu(10, 10);
        array d     = makeSparse<TypeParam>(dense, 5);
        s           = sparse(d);
    }

    // At this point only the sparse array will be allocated in memory.
    // Determine how much memory is allocated by one sparse array
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    size_t size_of_alloc      = lock_bytes;
    size_t buffers_per_sparse = lock_buffers;

    {
        array s2 = s.copy();
        s2.eval();

        // Make sure that the deep copy allocated additional memory
        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        EXPECT_NE(s.get(), s2.get()) << "The sparse arrays point to the same "
                                        "af_array object.";
        EXPECT_EQ(size_of_alloc * 2, lock_bytes)
            << "The number of bytes allocated by the deep copy do "
               "not match the original array";

        EXPECT_EQ(buffers_per_sparse * 2, lock_buffers)
            << "The number of buffers allocated by the deep "
               "copy do not match the original array";
        array d  = dense(s);
        array d2 = dense(s2);
        ASSERT_ARRAYS_EQ(d, d2);
    }
}

TYPED_TEST(Sparse, Empty) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    af_array ret = 0;
    dim_t rows = 0, cols = 0, nnz = 0;
    EXPECT_EQ(AF_SUCCESS, af_create_sparse_array_from_ptr(
                              &ret, rows, cols, nnz, NULL, NULL, NULL,
                              (af_dtype)dtype_traits<TypeParam>::af_type,
                              AF_STORAGE_CSR, afHost));
    bool sparse = false;
    EXPECT_EQ(AF_SUCCESS, af_is_sparse(&sparse, ret));
    EXPECT_EQ(true, sparse);
    EXPECT_EQ(AF_SUCCESS, af_release_array(ret));
}

TYPED_TEST(Sparse, EmptyDeepCopy) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    array a = sparse(0, 0, array(0, (af_dtype)dtype_traits<TypeParam>::af_type),
                     array(1, s32), array(0, s32));
    EXPECT_TRUE(a.issparse());
    EXPECT_EQ(0, sparseGetNNZ(a));

    array b = a.copy();
    EXPECT_TRUE(b.issparse());
    EXPECT_EQ(0, sparseGetNNZ(b));
}

TEST(Sparse, CPPSparseFromHostArrays) {
    //! [ex_sparse_host_arrays]

    float vals[]  = {5, 8, 3, 6};
    int row_ptr[] = {0, 0, 2, 3, 4};
    int col_idx[] = {0, 1, 2, 1};
    const int M = 4, N = 4, nnz = 4;

    // Create sparse array (CSR) from host pointers to values, row
    // pointers, and column indices.
    array sparse = af::sparse(M, N, nnz, vals, row_ptr, col_idx, f32,
                              AF_STORAGE_CSR, afHost);

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    //! [ex_sparse_host_arrays]

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    af::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, array(dim4(nnz, 1), vals));
    ASSERT_ARRAYS_EQ(sparse_row_ptr, array(dim4(M + 1, 1), row_ptr));
    ASSERT_ARRAYS_EQ(sparse_col_idx, array(dim4(nnz, 1), col_idx));
    ASSERT_EQ(sparse_storage, AF_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);
}

TEST(Sparse, CPPSparseFromAFArrays) {
    //! [ex_sparse_af_arrays]

    float v[]   = {5, 8, 3, 6};
    int r[]     = {0, 0, 2, 3, 4};
    int c[]     = {0, 1, 2, 1};
    const int M = 4, N = 4, nnz = 4;
    array vals    = array(dim4(nnz), v);
    array row_ptr = array(dim4(M + 1), r);
    array col_idx = array(dim4(nnz), c);

    // Create sparse array (CSR) from af::arrays containing values,
    // row pointers, and column indices.
    array sparse = af::sparse(M, N, vals, row_ptr, col_idx, AF_STORAGE_CSR);

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    //! [ex_sparse_af_arrays]

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    af::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, vals);
    ASSERT_ARRAYS_EQ(sparse_row_ptr, row_ptr);
    ASSERT_ARRAYS_EQ(sparse_col_idx, col_idx);
    ASSERT_EQ(sparse_storage, AF_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);
}

TEST(Sparse, CPPSparseFromDenseUsage) {
    float dns[] = {0, 5, 0, 0, 0, 8, 0, 6, 0, 0, 3, 0, 0, 0, 0, 0};
    const int M = 4, N = 4, nnz = 4;
    array dense(dim4(M, N), dns);

    //! [ex_sparse_from_dense]

    // dense
    //     0     0     0     0
    //     5     8     0     0
    //     0     0     3     0
    //     0     6     0     0

    // Convert dense af::array to its sparse (CSR) representation.
    array sparse = af::sparse(dense, AF_STORAGE_CSR);

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    //! [ex_sparse_from_dense]

    float v[] = {5, 8, 3, 6};
    int r[]   = {0, 0, 2, 3, 4};
    int c[]   = {0, 1, 2, 1};
    array gold_vals(dim4(nnz), v);
    array gold_row_ptr(dim4(M + 1), r);
    array gold_col_idx(dim4(nnz), c);

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    af::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, gold_vals);
    ASSERT_ARRAYS_EQ(sparse_row_ptr, gold_row_ptr);
    ASSERT_ARRAYS_EQ(sparse_col_idx, gold_col_idx);
    ASSERT_EQ(sparse_storage, AF_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);
}

TEST(Sparse, CPPDenseToSparseToDenseUsage) {
    float g[]   = {0, 5, 0, 0, 0, 8, 0, 6, 0, 0, 3, 0, 0, 0, 0, 0};
    const int M = 4, N = 4;
    array in(dim4(M, N), g);
    array sparse = af::sparse(in, AF_STORAGE_CSR);

    //! [ex_dense_from_sparse]

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    // Get dense representation of given sparse af::array.
    array dense = af::dense(sparse);

    // dense
    //     0     0     0     0
    //     5     8     0     0
    //     0     0     3     0
    //     0     6     0     0

    //! [ex_dense_from_sparse]

    float v[]     = {5, 8, 3, 6};
    int r[]       = {0, 0, 2, 3, 4};
    int c[]       = {0, 1, 2, 1};
    const int nnz = 4;
    array gold_vals(dim4(nnz), v);
    array gold_row_ptr(dim4(M + 1), r);
    array gold_col_idx(dim4(nnz), c);

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    af::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, gold_vals);
    ASSERT_ARRAYS_EQ(sparse_row_ptr, gold_row_ptr);
    ASSERT_ARRAYS_EQ(sparse_col_idx, gold_col_idx);
    ASSERT_EQ(sparse_storage, AF_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);

    // Check dense array
    array gold(dim4(M, N), g);
    ASSERT_ARRAYS_EQ(in, gold);
    ASSERT_ARRAYS_EQ(dense, gold);
}

TEST(Sparse, CPPDenseToSparseConversions) {
    array in      = af::randu(200, 200);
    in(in < 0.75) = 0;

    array coo_sparse_arr = af::sparse(in, AF_STORAGE_COO);
    array csr_sparse_arr = af::sparse(in, AF_STORAGE_CSR);

    array coo_dense_arr = af::dense(coo_sparse_arr);
    array csr_dense_arr = af::dense(csr_sparse_arr);

    ASSERT_ARRAYS_EQ(in, coo_dense_arr);
    ASSERT_ARRAYS_EQ(in, csr_dense_arr);

    array non_zero   = af::flat(in)(af::where(in));
    array non_zero_T = af::flat(in.T())(af::where(in.T()));
    ASSERT_ARRAYS_EQ(non_zero, af::sparseGetValues(coo_sparse_arr));
    ASSERT_ARRAYS_EQ(
        non_zero_T,
        af::sparseGetValues(csr_sparse_arr));  // csr values are transposed
}