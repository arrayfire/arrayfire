/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <sparse_common.hpp>

#define SPARSE_TESTS(T, eps)                                \
    TEST(SPARSE, T##Square)                                 \
    {                                                       \
        sparseTester<T>(1000, 1000, 100, 5, eps);           \
    }                                                       \
    TEST(SPARSE, T##RectMultiple)                           \
    {                                                       \
        sparseTester<T>(2048, 1024, 512, 3, eps);           \
    }                                                       \
    TEST(SPARSE, T##RectDense)                              \
    {                                                       \
        sparseTester<T>(500, 1000, 250, 1, eps);            \
    }                                                       \
    TEST(SPARSE, T##MatVec)                                 \
    {                                                       \
        sparseTester<T>(625, 1331, 1, 2, eps);              \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##MatVec)                       \
    {                                                       \
        sparseTransposeTester<T>(625, 1331, 1, 2, eps);     \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##Square)                       \
    {                                                       \
        sparseTransposeTester<T>(1000, 1000, 100, 5, eps);  \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##RectMultiple)                 \
    {                                                       \
        sparseTransposeTester<T>(2048, 1024, 512, 3, eps);  \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##RectDense)                    \
    {                                                       \
        sparseTransposeTester<T>(453, 751, 397, 1, eps);    \
    }                                                       \
    TEST(SPARSE, T##ConvertCSR)                             \
    {                                                       \
        convertCSR<T>(2345, 5678, 0.5);                     \
    }                                                       \

SPARSE_TESTS(float, 1E-3)
SPARSE_TESTS(double, 1E-5)
SPARSE_TESTS(cfloat, 1E-3)
SPARSE_TESTS(cdouble, 1E-5)

#undef SPARSE_TESTS

#define CREATE_TESTS(STYPE)                                         \
    TEST(SPARSE_CREATE, STYPE)                                      \
    {                                                               \
        createFunction<STYPE>();                                    \
    }

CREATE_TESTS(AF_STORAGE_CSR)
CREATE_TESTS(AF_STORAGE_COO)

#undef CREATE_TESTS

TEST(SPARSE_CREATE, AF_STORAGE_CSC)
{
    af::array d = af::identity(3, 3);

    af_array out = 0;
    ASSERT_EQ(AF_ERR_ARG, af_create_sparse_array_from_dense(&out, d.get(), AF_STORAGE_CSC));

    if(out != 0) af_release_array(out);
}

#define CAST_TESTS_TYPES(Ti, To, SUFFIX, M, N, F)                               \
    TEST(SPARSE_CAST, Ti##_##To##_##SUFFIX)                                     \
    {                                                                           \
        sparseCastTester<Ti, To>(M, N, F);                                      \
    }                                                                           \

#define CAST_TESTS(Ti, To)                                                      \
    CAST_TESTS_TYPES(Ti, To, 1, 1000, 1000,  5)                                 \
    CAST_TESTS_TYPES(Ti, To, 2,  512, 1024,  2)                                 \

CAST_TESTS(float  , float   )
CAST_TESTS(float  , double  )
CAST_TESTS(float  , cfloat  )
CAST_TESTS(float  , cdouble )

CAST_TESTS(double , float   )
CAST_TESTS(double , double  )
CAST_TESTS(double , cfloat  )
CAST_TESTS(double , cdouble )

CAST_TESTS(cfloat , cfloat  )
CAST_TESTS(cfloat , cdouble )

CAST_TESTS(cdouble, cfloat  )
CAST_TESTS(cdouble, cdouble )


TEST(Sparse, ISSUE_1745)
{
  af::array A = af::randu(4, 4);
  A(1, af::span) = 0;
  A(2, af::span) = 0;

  af::array idx = where(A);
  af::array data = A(idx);
  af::array row_idx = (idx / A.dims()[0]).as(s64);
  af::array col_idx = (idx % A.dims()[0]).as(s64);

  af_array A_sparse;
  ASSERT_EQ(AF_ERR_ARG, af_create_sparse_array(&A_sparse, A.dims(0), A.dims(1), data.get(), row_idx.get(), col_idx.get(), AF_STORAGE_CSR));
}

template<typename T>
class Sparse : public ::testing::Test {};

typedef ::testing::Types<float, af::cfloat, double, af::cdouble> SparseTypes;
TYPED_TEST_CASE(Sparse, SparseTypes);

TYPED_TEST(Sparse, DeepCopy) {
    if (noDoubleTests<TypeParam>()) return;
    using namespace af;
    cleanSlate();

    array s;
    {
        // Create a sparse array from a dense array. Make sure that the dense arrays
        // are removed
        array dense = randu(10, 10);
        array d = makeSparse<TypeParam>(dense, 5);
        s = sparse(d);
    }

    // At this point only the sparse array will be allocated in memory. Determine
    // how much memory is allocated by one sparse array
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);
    size_t size_of_alloc = lock_bytes;
    size_t buffers_per_sparse = lock_buffers;

    {
        array s2 = s.copy();
        s2.eval();

        // Make sure that the deep copy allocated additional memory
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        EXPECT_NE(s.get(), s2.get()) << "The sparse arrays point to the same "
                                        "af_array object.";
        EXPECT_EQ(size_of_alloc * 2,
                  lock_bytes) << "The number of bytes allocated by the deep copy do "
                                "not match the original array";

        EXPECT_EQ(buffers_per_sparse * 2,
                  lock_buffers) << "The number of buffers allocated by the deep "
                                  "copy do not match the original array";
        array d = dense(s);
        array d2 = dense(s2);
        ASSERT_TRUE(allTrue<bool>(d == d2));
    }
}
