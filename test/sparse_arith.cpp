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
using std::abs;
using af::array;
using af::cfloat;
using af::cdouble;
using af::deviceGC;
using af::dim4;
using af::freeHost;
using af::max;
using af::sum;

template<typename T>
array makeSparse(array A, int factor)
{
    A = floor(A * 1000);
    A = A * ((A % factor) == 0) / 1000;
    return A;
}

template<>
array makeSparse<cfloat>(array A, int factor)
{
    array r = real(A);
    r = floor(r * 1000);
    r = r * ((r % factor) == 0) / 1000;

    array i = r / 2;

    A = complex(r, i);
    return A;
}

template<>
array makeSparse<cdouble>(array A, int factor)
{
    array r = real(A);
    r = floor(r * 1000);
    r = r * ((r % factor) == 0) / 1000;

    array i = r / 2;

    A = complex(r, i);
    return A;
}

typedef enum {
    af_add_t,
    af_sub_t,
    af_mul_t,
    af_div_t,
} af_op_t;

template<af_op_t op>
struct arith_op
{
    array operator()(array v1, array v2)
    {
        return v1;
    }
};

template<>
struct arith_op<af_add_t>
{
    array operator()(array v1, array v2)
    {
        return v1 + v2;
    }
};

template<>
struct arith_op<af_sub_t>
{
    array operator()(array v1, array v2)
    {
        return v1 - v2;
    }
};

template<>
struct arith_op<af_mul_t>
{
    array operator()(array v1, array v2)
    {
        return v1 * v2;
    }
};

template<>
struct arith_op<af_div_t>
{
    array operator()(array v1, array v2)
    {
        return v1 / v2;
    }
};

template<typename T>
void sparseCompare(array A, array B, const double eps)
{
// This macro is used to check if either value is finite and then call assert
// If neither value is finite, then they can be assumed to be equal to either inf or nan
#define ASSERT_FINITE_EQ(V1, V2)                    \
    if(std::isfinite(V1) || std::isfinite(V2)) {    \
        ASSERT_NEAR(V1, V2, eps) << "at : " << i;   \
    }                                               \

    array AValues = sparseGetValues(A);
    array ARowIdx = sparseGetRowIdx(A);
    array AColIdx = sparseGetColIdx(A);

    array BValues = sparseGetValues(B);
    array BRowIdx = sparseGetRowIdx(B);
    array BColIdx = sparseGetColIdx(B);

    // Verify row and col indices
    ASSERT_EQ(0, max<int>(ARowIdx - BRowIdx));
    ASSERT_EQ(0, max<int>(AColIdx - BColIdx));

    T *ptrA = AValues.host<T>();
    T *ptrB = BValues.host<T>();
    for(int i = 0; i < AValues.elements(); i++) {
        ASSERT_FINITE_EQ(real(ptrA[i]), real(ptrB[i]));

        if(A.iscomplex()) {
            ASSERT_FINITE_EQ(imag(ptrA[i]), imag(ptrB[i]));
        }
    }
    freeHost(ptrA);
    freeHost(ptrB);

#undef ASSERT_FINITE_EQ
}

template<typename T, af_op_t op>
void sparseArithTester(const int m, const int n, int factor, const double eps)
{
    deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    array A = cpu_randu<T>(dim4(m, n));
    array B = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::af_type);
    array B = randu(m, n, (dtype)dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    array RA = sparse(A, AF_STORAGE_CSR);
    array OA = sparse(A, AF_STORAGE_COO);

    // Arith Op
    array resR = arith_op<op>()(RA, B);
    array resO = arith_op<op>()(OA, B);
    array resD = arith_op<op>()( A, B);

    array revR = arith_op<op>()(B, RA);
    array revO = arith_op<op>()(B, OA);
    array revD = arith_op<op>()(B,  A);

    ASSERT_NEAR(0, sum<double>(abs(real(resR - resD))) / (m * n), eps);
    ASSERT_NEAR(0, sum<double>(abs(imag(resR - resD))) / (m * n), eps);

    ASSERT_NEAR(0, sum<double>(abs(real(resO - resD))) / (m * n), eps);
    ASSERT_NEAR(0, sum<double>(abs(imag(resO - resD))) / (m * n), eps);

    ASSERT_NEAR(0, sum<double>(abs(real(revR - revD))) / (m * n), eps);
    ASSERT_NEAR(0, sum<double>(abs(imag(revR - revD))) / (m * n), eps);

    ASSERT_NEAR(0, sum<double>(abs(real(revO - revD))) / (m * n), eps);
    ASSERT_NEAR(0, sum<double>(abs(imag(revO - revD))) / (m * n), eps);
}

// Mul
template<typename T>
void sparseArithTesterMul(const int m, const int n, int factor, const double eps)
{
    deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    array A = cpu_randu<T>(dim4(m, n));
    array B = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::af_type);
    array B = randu(m, n, (dtype)dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    array RA = sparse(A, AF_STORAGE_CSR);
    array OA = sparse(A, AF_STORAGE_COO);

    // Forward
    {
        // Arith Op
        array resR = arith_op<af_mul_t>()(RA, B);
        array resO = arith_op<af_mul_t>()(OA, B);

        // We will test this by converting the COO to CSR and CSR to COO and
        // comparing them. In essense, we are comparing the resR and resO
        // TODO: Make a better comparison using dense

        // Check resR against conR
        array conR = sparseConvertTo(resR, AF_STORAGE_CSR);
        sparseCompare<T>(resR, conR, eps);

        // Check resO against conO
        array conO = sparseConvertTo(resR, AF_STORAGE_COO);
        sparseCompare<T>(resO, conO, eps);
    }

    // Reverse
    {
        // Arith Op
        array resR = arith_op<af_mul_t>()(B, RA);
        array resO = arith_op<af_mul_t>()(B, OA);

        // We will test this by converting the COO to CSR and CSR to COO and
        // comparing them. In essense, we are comparing the resR and resO
        // TODO: Make a better comparison using dense

        // Check resR against conR
        array conR = sparseConvertTo(resR, AF_STORAGE_CSR);
        sparseCompare<T>(resR, conR, eps);

        // Check resO against conO
        array conO = sparseConvertTo(resR, AF_STORAGE_COO);
        sparseCompare<T>(resO, conO, eps);
    }
}

// Div
template<typename T>
void sparseArithTesterDiv(const int m, const int n, int factor, const double eps)
{
    deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    array A = cpu_randu<T>(dim4(m, n));
    array B = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::af_type);
    array B = randu(m, n, (dtype)dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    array RA = sparse(A, AF_STORAGE_CSR);
    array OA = sparse(A, AF_STORAGE_COO);

    // Arith Op
    array resR = arith_op<af_div_t>()(RA, B);
    array resO = arith_op<af_div_t>()(OA, B);

    // Assert division by sparse is not allowed
    af_array out_temp = 0;
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED, af_div(&out_temp, B.get(), RA.get(), false));
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED, af_div(&out_temp, B.get(), OA.get(), false));
    if(out_temp != 0) af_release_array(out_temp);

    // We will test this by converting the COO to CSR and CSR to COO and
    // comparing them. In essense, we are comparing the resR and resO
    // TODO: Make a better comparison using dense

    // Check resR against conR
    array conR = sparseConvertTo(resR, AF_STORAGE_CSR);
    sparseCompare<T>(resR, conR, eps);

    // Check resO against conO
    array conO = sparseConvertTo(resR, AF_STORAGE_COO);
    sparseCompare<T>(resO, conO, eps);
}

#define ARITH_TESTS_OPS(T, M, N, F, EPS)                                    \
    TEST(SPARSE_ARITH, T##_ADD_##M##_##N)                                   \
    {                                                                       \
        sparseArithTester<T, af_add_t>(M, N, F, EPS);                       \
    }                                                                       \
    TEST(SPARSE_ARITH, T##_SUB_##M##_##N)                                   \
    {                                                                       \
        sparseArithTester<T, af_sub_t>(M, N, F, EPS);                       \
    }                                                                       \
    TEST(SPARSE_ARITH, T##_MUL_##M##_##N)                                   \
    {                                                                       \
        sparseArithTesterMul<T>(M, N, F, EPS);                              \
    }                                                                       \
    TEST(SPARSE_ARITH, T##_DIV_##M##_##N)                                   \
    {                                                                       \
        sparseArithTesterDiv<T>(M, N, F, EPS);                              \
    }                                                                       \

#define ARITH_TESTS(T, eps)                                                 \
    ARITH_TESTS_OPS(T, 10  , 10  , 5, eps)                                  \
    ARITH_TESTS_OPS(T, 1024, 1024, 5, eps)                                  \
    ARITH_TESTS_OPS(T, 100 , 100 , 1, eps)                                  \
    ARITH_TESTS_OPS(T, 2048, 1000, 6, eps)                                  \
    ARITH_TESTS_OPS(T, 123 , 278 , 5, eps)                                  \

ARITH_TESTS(float  , 1e-6)
ARITH_TESTS(double , 1e-6)
ARITH_TESTS(cfloat , 1e-4) // This is mostly for complex division in OpenCL
ARITH_TESTS(cdouble, 1e-6)
