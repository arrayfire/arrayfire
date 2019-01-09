/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::max;
using std::abs;
using std::string;
using std::vector;

///////////////////////////////// CPP ////////////////////////////////////
//

template <typename T>
array makeSparse(array A, int factor) {
    A = floor(A * 1000);
    A = A * ((A % factor) == 0) / 1000;
    return A;
}

template <>
array makeSparse<cfloat>(array A, int factor) {
    array r = real(A);
    r       = floor(r * 1000);
    r       = r * ((r % factor) == 0) / 1000;

    array i = r / 2;

    A = complex(r, i);
    return A;
}

template <>
array makeSparse<cdouble>(array A, int factor) {
    array r = real(A);
    r       = floor(r * 1000);
    r       = r * ((r % factor) == 0) / 1000;

    array i = r / 2;

    A = complex(r, i);
    return A;
}

template <typename T, af_storage src, af_storage dest>
void sparseConvertTester(const int m, const int n, int factor) {
    if (noDoubleTests<T>()) return;

    array A = cpu_randu<T>(dim4(m, n));

    A = makeSparse<T>(A, factor);

    // Create Sparse Array of type src and dest From Dense
    array sA = sparse(A, src);

    // Convert src to dest format and dest to src
    array s2d = sparseConvertTo(sA, dest);

    // Create the dest type from dense - gold
    array dA = sparse(A, dest);

    // Verify nnZ
    dim_t dNNZ   = sparseGetNNZ(dA);
    dim_t s2dNNZ = sparseGetNNZ(s2d);

    ASSERT_EQ(dNNZ, s2dNNZ);

    // Verify Types
    af_storage dType   = sparseGetStorage(dA);
    af_storage s2dType = sparseGetStorage(s2d);

    ASSERT_EQ(dType, s2dType);

    // Get the individual arrays and verify equality
    array dValues = sparseGetValues(dA);
    array dRowIdx = sparseGetRowIdx(dA);
    array dColIdx = sparseGetColIdx(dA);

    array s2dValues = sparseGetValues(s2d);
    array s2dRowIdx = sparseGetRowIdx(s2d);
    array s2dColIdx = sparseGetColIdx(s2d);

    // Verify values
    ASSERT_EQ(0, max<double>(real(dValues - s2dValues)));
    ASSERT_EQ(0, max<double>(imag(dValues - s2dValues)));

    // Verify row and col indices
    ASSERT_EQ(0, max<int>(dRowIdx - s2dRowIdx));
    ASSERT_EQ(0, max<int>(dColIdx - s2dColIdx));
}

#define CONVERT_TESTS_TYPES(T, STYPE, DTYPE, SUFFIX, M, N, F) \
    TEST(SPARSE_CONVERT, T##_##STYPE##_##DTYPE##_##SUFFIX) {  \
        sparseConvertTester<T, STYPE, DTYPE>(M, N, F);        \
    }                                                         \
    TEST(SPARSE_CONVERT, T##_##DTYPE##_##STYPE##_##SUFFIX) {  \
        sparseConvertTester<T, DTYPE, STYPE>(M, N, F);        \
    }

#define CONVERT_TESTS(T, STYPE, DTYPE)                      \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 1, 1000, 1000, 5)  \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 2, 512, 512, 1)    \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 3, 512, 1024, 2)   \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 4, 2048, 1024, 10) \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 5, 237, 411, 5)

CONVERT_TESTS(float, AF_STORAGE_CSR, AF_STORAGE_COO)
CONVERT_TESTS(double, AF_STORAGE_CSR, AF_STORAGE_COO)
CONVERT_TESTS(cfloat, AF_STORAGE_CSR, AF_STORAGE_COO)
CONVERT_TESTS(cdouble, AF_STORAGE_CSR, AF_STORAGE_COO)

#undef CONVERT_TESTS
#undef CONVERT_TESTS_TYPES

// Test to check failure with CSC
TEST(SPARSE_CONVERT, CSC_ARG_ERROR) {
    const int m = 100, n = 28, factor = 5;

    array A = cpu_randu<float>(dim4(m, n));

    A = makeSparse<float>(A, factor);

    // Create Sparse Array of type src and dest From Dense
    array sA = sparse(A, AF_STORAGE_CSR);

    // Convert src to dest format and dest to src
    // Use C-API to catch error
    af_array out = 0;
    ASSERT_EQ(AF_ERR_ARG, af_sparse_convert_to(&out, sA.get(), AF_STORAGE_CSC));

    if (out != 0) af_release_array(out);
}
