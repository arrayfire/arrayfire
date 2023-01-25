/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <arrayfire.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::cdouble;
using af::cfloat;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

///////////////////////////////// CPP ////////////////////////////////////
//

template<typename T>
static af::array makeSparse(af::array A, int factor) {
    A = floor(A * 1000);
    A = A * ((A % factor) == 0) / 1000;
    return A;
}

template<>
af::array makeSparse<cfloat>(af::array A, int factor) {
    af::array r = real(A);
    r           = floor(r * 1000);
    r           = r * ((r % factor) == 0) / 1000;

    af::array i = r / 2;

    A = af::complex(r, i);
    return A;
}

template<>
af::array makeSparse<cdouble>(af::array A, int factor) {
    af::array r = real(A);
    r           = floor(r * 1000);
    r           = r * ((r % factor) == 0) / 1000;

    af::array i = r / 2;

    A = af::complex(r, i);
    return A;
}

static double calc_norm(af::array lhs, af::array rhs) {
    return af::max<double>(af::abs(lhs - rhs) /
                           (af::abs(lhs) + af::abs(rhs) + 1E-5));
}

template<typename T>
static void sparseTester(const int m, const int n, const int k, int factor,
                         double eps, int targetDevice = -1) {
    if (targetDevice >= 0) af::setDevice(targetDevice);

    af::deviceGC();

    SUPPORTED_TYPE_CHECK(T);

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
static void sparseTransposeTester(const int m, const int n, const int k,
                                  int factor, double eps,
                                  int targetDevice = -1) {
    if (targetDevice >= 0) af::setDevice(targetDevice);

    af::deviceGC();

    SUPPORTED_TYPE_CHECK(T);

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
    af::array dRes3;
    if (IsComplex<T>::value) {
        dRes3 = matmul(A, B, AF_MAT_CTRANS, AF_MAT_NONE);
    }

    // Create Sparse Array From Dense
    af::array sA = af::sparse(A, AF_STORAGE_CSR);

    // Sparse Matmul
    af::array sRes2 = matmul(sA, B, AF_MAT_TRANS, AF_MAT_NONE);
    af::array sRes3;
    if (IsComplex<T>::value) {
        sRes3 = matmul(sA, B, AF_MAT_CTRANS, AF_MAT_NONE);
    }

    // Verify Results
    ASSERT_NEAR(0, calc_norm(real(dRes2), real(sRes2)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes2), imag(sRes2)), eps);

    if (IsComplex<T>::value) {
        ASSERT_NEAR(0, calc_norm(real(dRes3), real(sRes3)), eps);
        ASSERT_NEAR(0, calc_norm(imag(dRes3), imag(sRes3)), eps);
    }
}

template<typename T>
static void convertCSR(const int M, const int N, const double ratio,
                       int targetDevice = -1) {
    if (targetDevice >= 0) af::setDevice(targetDevice);

    SUPPORTED_TYPE_CHECK(T);
#if 1
    af::array a = cpu_randu<T>(af::dim4(M, N));
#else
    af::array a = af::randu(M, N);
#endif
    a = a * (a > ratio);

    af::array s  = af::sparse(a, AF_STORAGE_CSR);
    af::array aa = af::dense(s);

    ASSERT_ARRAYS_EQ(a, aa);
}

template<typename T>
static void convertCSC(const int M, const int N, const double ratio,
                       int targetDevice = -1) {
    if (targetDevice >= 0) af::setDevice(targetDevice);

    SUPPORTED_TYPE_CHECK(T);
#if 1
    af::array a = cpu_randu<T>(af::dim4(M, N));
#else
    af::array a = af::randu(M, N);
#endif
    a = a * (a > ratio);

    af::array s  = af::sparse(a, AF_STORAGE_CSC);
    af::array aa = af::dense(s);

    ASSERT_ARRAYS_EQ(a, aa);
}

// This test essentially verifies that the sparse structures have the correct
// dimensions and indices using a very basic test
template<af_storage stype>
static void createFunction() {
    af::array in = af::sparse(af::identity(3, 3), stype);

    af::array values = sparseGetValues(in);
    af::array rowIdx = sparseGetRowIdx(in);
    af::array colIdx = sparseGetColIdx(in);
    dim_t nNZ        = sparseGetNNZ(in);

    ASSERT_EQ(nNZ, values.elements());

    ASSERT_EQ(0, af::max<double>(values - af::constant(1, nNZ)));
    ASSERT_EQ(0, af::max<int>(rowIdx -
                              af::range(af::dim4(rowIdx.elements()), 0, s32)));
    ASSERT_EQ(0, af::max<int>(colIdx -
                              af::range(af::dim4(colIdx.elements()), 0, s32)));
}

template<typename Ti, typename To>
static void sparseCastTester(const int m, const int n, int factor) {
    SUPPORTED_TYPE_CHECK(Ti);
    SUPPORTED_TYPE_CHECK(To);

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
    if (iValues.iscomplex() && !oValues.iscomplex()) {
        ASSERT_NEAR(0, af::max<double>(af::abs(af::abs(iValues) - oValues)),
                    eps);
    } else if (!iValues.iscomplex() && oValues.iscomplex()) {
        ASSERT_NEAR(0, af::max<double>(af::abs(iValues - af::abs(oValues))),
                    eps);
    } else {
        ASSERT_NEAR(0, af::max<double>(af::abs(iValues - oValues)), eps);
    }
}
