/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::abs;
using af::cfloat;
using af::cdouble;

///////////////////////////////// CPP ////////////////////////////////////
//

template<typename T>
af::array makeSparse(af::array A, int factor)
{
    A = floor(A * 1000);
    A = A * ((A % factor) == 0) / 1000;
    return A;
}

template<>
af::array makeSparse<cfloat>(af::array A, int factor)
{
    af::array r = real(A);
    r = floor(r * 1000);
    r = r * ((r % factor) == 0) / 1000;

    af::array i = r / 2;

    A = af::complex(r, i);
    return A;
}

template<>
af::array makeSparse<cdouble>(af::array A, int factor)
{
    af::array r = real(A);
    r = floor(r * 1000);
    r = r * ((r % factor) == 0) / 1000;

    af::array i = r / 2;

    A = af::complex(r, i);
    return A;
}

double calc_norm(af::array lhs, af::array rhs)
{
    return af::max<double>(af::abs(lhs - rhs) / (af::abs(lhs) + af::abs(rhs) + 1E-5));
}

template<typename T>
void sparseTester(const int m, const int n, const int k, int factor, double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(n, k));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(n, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    // Result of GEMM
    af::array dRes1 = matmul(A, B);

    // Create Sparse Array From Dense
    af::array sA = af::sparse(A, AF_STORAGE_CSR);

    // Sparse Matmul
    af::array sRes1 = matmul(sA, B);

    // Verify Results
    ASSERT_NEAR(0, calc_norm(real(dRes1), real(sRes1)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes1), imag(sRes1)), eps);
}

template<typename T>
void sparseTransposeTester(const int m, const int n, const int k, int factor, double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(m, k));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(m, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    // Result of GEMM
    af::array dRes2 = matmul(A, B, AF_MAT_TRANS, AF_MAT_NONE);
    af::array dRes3 = matmul(A, B, AF_MAT_CTRANS, AF_MAT_NONE);

    // Create Sparse Array From Dense
    af::array sA = af::sparse(A, AF_STORAGE_CSR);

    // Sparse Matmul
    af::array sRes2 = matmul(sA, B, AF_MAT_TRANS, AF_MAT_NONE);
    af::array sRes3 = matmul(sA, B, AF_MAT_CTRANS, AF_MAT_NONE);

    // Verify Results
    ASSERT_NEAR(0, calc_norm(real(dRes2), real(sRes2)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes2), imag(sRes2)), eps);

    ASSERT_NEAR(0, calc_norm(real(dRes3), real(sRes3)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes3), imag(sRes3)), eps);
}

template<typename T>
void convertCSR(const int M, const int N, const float ratio)
{
    if (noDoubleTests<T>()) return;
#if 1
    af::array a = cpu_randu<T>(af::dim4(M, N));
#else
    af::array a = af::randu(M, N);
#endif
    a = a * (a > ratio);

    af::array s = af::sparse(a, AF_STORAGE_CSR);
    af::array aa = af::dense(s);

    ASSERT_EQ(0, af::max<double>(af::abs(a - aa)));
}

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

// This test essentially verifies that the sparse structures have the correct
// dimensions and indices using a very basic test
template<af_storage stype>
void createFunction()
{
    af::array in = af::sparse(af::identity(3, 3), stype);

    af::array values = sparseGetValues(in);
    af::array rowIdx = sparseGetRowIdx(in);
    af::array colIdx = sparseGetColIdx(in);
    dim_t     nNZ    = sparseGetNNZ(in);

    ASSERT_EQ(nNZ, values.elements());

    ASSERT_EQ(0, af::max<double>(values - af::constant(1, nNZ)));
    ASSERT_EQ(0, af::max<int   >(rowIdx - af::range(af::dim4(rowIdx.elements()), 0, s32)));
    ASSERT_EQ(0, af::max<int   >(colIdx - af::range(af::dim4(colIdx.elements()), 0, s32)));
}

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

template<typename Ti, typename To>
void sparseCastTester(const int m, const int n, int factor)
{
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

    af::array A = cpu_randu<Ti>(af::dim4(m, n));

    A = makeSparse<Ti>(A, factor);

    af::array sTi = af::sparse(A, AF_STORAGE_CSR);

    // Cast
    af::array sTo = sTi.as((af::dtype)af::dtype_traits<To>::af_type);

    // Verify nnZ
    dim_t iNNZ = sparseGetNNZ(sTi);
    dim_t oNNZ = sparseGetNNZ(sTo);

    ASSERT_EQ(iNNZ, oNNZ);

    // Verify Types
    dim_t iSType = sparseGetStorage(sTi);
    dim_t oSType = sparseGetStorage(sTo);

    ASSERT_EQ(iSType, oSType);

    // Get the individual arrays and verify equality
    af::array iValues = sparseGetValues(sTi);
    af::array iRowIdx = sparseGetRowIdx(sTi);
    af::array iColIdx = sparseGetColIdx(sTi);

    af::array oValues = sparseGetValues(sTo);
    af::array oRowIdx = sparseGetRowIdx(sTo);
    af::array oColIdx = sparseGetColIdx(sTo);

    // Verify values
    ASSERT_EQ(0, af::max<int>(af::abs(iRowIdx - oRowIdx)));
    ASSERT_EQ(0, af::max<int>(af::abs(iColIdx - oColIdx)));

    static const double eps = 1e-6;
    if(iValues.iscomplex() && !oValues.iscomplex()) {
        ASSERT_NEAR(0, af::max<double>(af::abs(af::abs(iValues) - oValues)), eps);
    } else if(!iValues.iscomplex() && oValues.iscomplex()) {
        ASSERT_NEAR(0, af::max<double>(af::abs(iValues - af::abs(oValues))), eps);
    } else {
        ASSERT_NEAR(0, af::max<double>(af::abs(iValues - oValues)), eps);
    }
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
