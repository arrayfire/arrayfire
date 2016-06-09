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

    af::array i = real(A);
    i = floor(i * 1000);
    i = i * ((i % factor) == 0) / 1000;

    A = af::complex(r, i);
    return A;
}

template<>
af::array makeSparse<cdouble>(af::array A, int factor)
{
    af::array r = real(A);
    r = floor(r * 1000);
    r = r * ((r % factor) == 0) / 1000;

    af::array i = real(A);
    i = floor(i * 1000);
    i = i * ((i % factor) == 0) / 1000;

    A = af::complex(r, i);
    return A;
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
    af::array dRes = matmul(A, B);

    // Create Sparse Array From Dense
    af::array sA = af::createSparseArray(A, AF_STORAGE_CSR);

    // Sparse Matmul
    af::array sRes = matmul(sA, B);

    // Verify Results
    ASSERT_NEAR(0, af::sum<double>(af::abs(real(dRes - sRes))) / (m * k), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(dRes - sRes))) / (m * k), eps);
}


#define SPARSE_TESTS(T, eps)                            \
    TEST(SPARSE, T##Square)                             \
    {                                                   \
        sparseTester<T>(1000, 1000, 100, 5, eps);       \
    }                                                   \
    TEST(SPARSE, T##RectMultiple)                       \
    {                                                   \
        sparseTester<T>(2048, 1024, 512, 3, eps);       \
    }                                                   \
    TEST(SPARSE, T##RectDense)                          \
    {                                                   \
        sparseTester<T>(500, 1000, 250, 1, eps);        \
    }                                                   \

SPARSE_TESTS(float, 0.01)
SPARSE_TESTS(double, 1E-5)
SPARSE_TESTS(cfloat, 0.01)
SPARSE_TESTS(cdouble, 1E-5)

#undef SPARSE_TESTS
