/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <complex>
#include <gtest/gtest.h>
#include <af/arith.h>
#include <af/data.h>
#include <testHelpers.hpp>

// This makes the macros cleaner
using std::abs;
using std::endl;
using std::vector;
using af::array;
using af::dtype_traits;
using af::exception;
using af::randu;

const int num = 10000;
const float flt_err = 1e-3;
const double dbl_err = 1e-10;

typedef std::complex<float> complex_float;
typedef std::complex<double> complex_double;

template<typename T>
T sigmoid(T in)
{
    return 1.0 / (1.0 + std::exp(-in));
}

#define TEST_REAL(T, func, err, lo, hi)                             \
    TEST(MathTests, Test_##func##_##T)                              \
    {                                                               \
        try {                                                       \
            if (noDoubleTests<T>()) return;                         \
            af_dtype ty = (af_dtype)dtype_traits<T>::af_type;       \
            array a = (hi - lo) * randu(num, ty) + lo + err;        \
            eval(a);                                                \
            array b = func(a);                                      \
            vector<T> h_a(a.elements());                            \
            vector<T> h_b(b.elements());                            \
            a.host(&h_a[0]);                                        \
            b.host(&h_b[0]);                                        \
                                                                    \
            for (int i = 0; i < num; i++) {                         \
                ASSERT_NEAR(h_b[i], func(h_a[i]), err) <<           \
                    "for value: " << h_a[i] << endl;                \
            }                                                       \
        } catch (exception &ex) {                                   \
            FAIL() << ex.what();                                    \
        }                                                           \
    }                                                               \

#define TEST_CPLX(T, func, err, lo, hi)                             \
    TEST(MathTests, Test_##func##_##T)                              \
    {                                                               \
        try {                                                       \
            if (noDoubleTests<T>()) return;                         \
            af_dtype ty = (af_dtype)dtype_traits<T>::af_type;       \
            array a = (hi - lo) * randu(num, ty) + lo + err;        \
            eval(a);                                                \
            array b = func(a);                                      \
            vector<T> h_a(a.elements());                            \
            vector<T> h_b(b.elements());                            \
            a.host(&h_a[0]);                                        \
            b.host(&h_b[0]);                                        \
                                                                    \
            for (int i = 0; i < num; i++) {                         \
                T res = func(h_a[i]);                               \
                ASSERT_NEAR(real(h_b[i]), real(res), err) <<        \
                    "for real value: " << h_a[i] << endl;           \
                ASSERT_NEAR(imag(h_b[i]), imag(res), err) <<        \
                    "for imag value: " << h_a[i] << endl;           \
            }                                                       \
        } catch (exception &ex) {                                   \
            FAIL() << ex.what();                                    \
        }                                                           \
    }                                                               \

#define MATH_TESTS_FLOAT(func) TEST_REAL(float, func, flt_err, 0.05f, 0.95f)
#define MATH_TESTS_DOUBLE(func) TEST_REAL(double, func, dbl_err, 0.05, 0.95)

#define MATH_TESTS_CFLOAT(func) TEST_CPLX(complex_float, func, flt_err, 0.05f, 0.95f)
#define MATH_TESTS_CDOUBLE(func) TEST_CPLX(complex_double, func, dbl_err, 0.05, 0.95)

#define MATH_TESTS_REAL(func)                   \
    MATH_TESTS_FLOAT(func)                      \
    MATH_TESTS_DOUBLE(func)                     \

#define MATH_TESTS_CPLX(func)                   \
    MATH_TESTS_CFLOAT(func)                     \
    MATH_TESTS_CDOUBLE(func)                    \

#define MATH_TESTS_ALL(func)                    \
    MATH_TESTS_REAL(func)                       \
    MATH_TESTS_CPLX(func)                       \

#define MATH_TESTS_LIMITS_REAL(func, lo, hi)    \
    TEST_REAL(float, func, flt_err, lo, hi)     \
    TEST_REAL(double, func, dbl_err, lo, hi)    \

#define MATH_TESTS_LIMITS_CPLX(func, lo, hi)            \
    TEST_CPLX(complex_float, func, flt_err, lo, hi)     \
    TEST_CPLX(complex_double, func, dbl_err, lo, hi)    \

MATH_TESTS_ALL(sin)
MATH_TESTS_ALL(cos)
MATH_TESTS_ALL(tan)

MATH_TESTS_REAL(asin)
MATH_TESTS_REAL(acos)
MATH_TESTS_REAL(atan)

MATH_TESTS_ALL(sinh)
MATH_TESTS_ALL(cosh)
MATH_TESTS_ALL(tanh)

MATH_TESTS_ALL(sqrt)
MATH_TESTS_ALL(exp)
MATH_TESTS_ALL(log)
MATH_TESTS_REAL(log10)
MATH_TESTS_REAL(log2)

MATH_TESTS_REAL(sigmoid)

MATH_TESTS_LIMITS_REAL(abs, -10, 10)
MATH_TESTS_LIMITS_REAL(ceil, -10, 10)
MATH_TESTS_LIMITS_REAL(floor, -10, 10)

#if __cplusplus > 199711L

MATH_TESTS_CPLX(asin)
MATH_TESTS_CPLX(acos)
MATH_TESTS_CPLX(atan)

MATH_TESTS_ALL(asinh)
MATH_TESTS_ALL(atanh)
MATH_TESTS_LIMITS_REAL(acosh, 1, 5)
MATH_TESTS_LIMITS_CPLX(acosh, 1, 5)
MATH_TESTS_LIMITS_REAL(round, -10, 10)
MATH_TESTS_REAL(cbrt)
MATH_TESTS_REAL(expm1)
MATH_TESTS_REAL(log1p)
MATH_TESTS_REAL(erf)
MATH_TESTS_REAL(erfc)
#endif

TEST(MathTests, Not)
{
    array a = randu(5, 5, b8);
    array b = !a;
    char *ha = a.host<char>();
    char *hb = b.host<char>();

    for(int i = 0; i < a.elements(); i++) {
        ASSERT_EQ(ha[i] ^ hb[i], true);
    }

    af_free_host(ha);
    af_free_host(hb);
}
