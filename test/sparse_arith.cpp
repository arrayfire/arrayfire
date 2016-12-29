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

typedef enum {
    af_add_t,
    af_sub_t,
    af_mul_t,
    af_div_t,
} af_op_t;

template<af_op_t op>
struct arith_op
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1;
    }
};

template<>
struct arith_op<af_add_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 + v2;
    }
};

template<>
struct arith_op<af_sub_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 - v2;
    }
};

template<>
struct arith_op<af_mul_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 * v2;
    }
};

template<>
struct arith_op<af_div_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 / v2;
    }
};

template<typename T, af_op_t op>
void sparseArithTester(const int m, const int n, int factor, const double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(m, n));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    af::array SA = af::sparse(A, AF_STORAGE_CSR);
    af::array OA = af::sparse(A, AF_STORAGE_COO);

    // Arith Op
    af::array resS = arith_op<op>()(SA, B);
    af::array resO = arith_op<op>()(OA, B);
    af::array resD = arith_op<op>()( A, B);

    af::array revS = arith_op<op>()(B, SA);
    af::array revO = arith_op<op>()(B, OA);
    af::array revD = arith_op<op>()(B,  A);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(resS - resD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(resS - resD))) / (m * n), eps);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(resO - resD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(resO - resD))) / (m * n), eps);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(revS - revD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(revS - revD))) / (m * n), eps);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(revO - revD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(revO - revD))) / (m * n), eps);
}

template<typename T>
void sparseArithTesterDiv(const int m, const int n, int factor, const double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(m, n));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    af::array SA = af::sparse(A, AF_STORAGE_CSR);
    af::array OA = af::sparse(A, AF_STORAGE_COO);

    // Arith Op
    af::array resS = arith_op<af_div_t>()(SA, B);
    af::array resO = arith_op<af_div_t>()(OA, B);
    af::array resD = arith_op<af_div_t>()( A, B);

    af::array revS = arith_op<af_div_t>()(B, SA);
    af::array revO = arith_op<af_div_t>()(B, OA);
    af::array revD = arith_op<af_div_t>()(B,  A);

    T *hResS = resS.host<T>();
    T *hResO = resO.host<T>();
    T *hResD = resD.host<T>();
    T *hRevS = revS.host<T>();
    T *hRevO = revO.host<T>();
    T *hRevD = revD.host<T>();

// This macro is used to check if either value is finite and then call assert
// If neither value is finite, then they can be assumed to be equal to either inf or nan
#define ASSERT_FINITE_EQ(V1, V2)                                                            \
    if(std::isfinite(V1) || std::isfinite(V2)) ASSERT_NEAR(V1, V2, eps) << "at : " << i;    \

    for(int i = 0; i < B.elements(); i++) {
        ASSERT_FINITE_EQ(real(hResS[i]), real(hResD[i]));
        ASSERT_FINITE_EQ(real(hResO[i]), real(hResD[i]));
        ASSERT_FINITE_EQ(real(hRevS[i]), real(hRevD[i]));
        ASSERT_FINITE_EQ(real(hRevO[i]), real(hRevD[i]));

        if(A.iscomplex()) {
            ASSERT_FINITE_EQ(imag(hResS[i]), imag(hResD[i]));
            ASSERT_FINITE_EQ(imag(hResO[i]), imag(hResD[i]));
            ASSERT_FINITE_EQ(imag(hRevS[i]), imag(hRevD[i]));
            ASSERT_FINITE_EQ(imag(hRevO[i]), imag(hRevD[i]));
        }
    }
#undef ASSERT_FINITE_EQ

    af::freeHost(hResS);
    af::freeHost(hResO);
    af::freeHost(hResD);
    af::freeHost(hRevS);
    af::freeHost(hRevO);
    af::freeHost(hRevD);
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
        sparseArithTester<T, af_mul_t>(M, N, F, EPS);                       \
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
