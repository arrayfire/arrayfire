/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <af/arith.h>
#include <af/data.h>
#include <testHelpers.hpp>

using namespace std;
using namespace af;

const int num = 10000;
const float flt_err = 1e-3;
const double dbl_err = 1e-10;

#define MATH_TESTS_LIMITS(Ti, To, func, err, lo, hi)            \
    TEST(MathTests, Test_##func##_##Ti)                         \
    {                                                           \
        if (noDoubleTests<Ti>()) return;                        \
        af_dtype ty = (af_dtype)dtype_traits<Ti>::af_type;      \
        af::array a = (hi - lo) * randu(num, ty) + lo + err;    \
        af::eval(a);                                            \
        af::array b = af::func(a);                              \
        Ti *h_a = a.host<Ti>();                                 \
        To *h_b = b.host<To>();                                 \
                                                                \
        for (int i = 0; i < num; i++)                           \
            ASSERT_NEAR(h_b[i], func(h_a[i]), err) <<           \
                "for value: " << h_a[i] << std::endl;           \
        delete[] h_a;                                           \
        delete[] h_b;                                           \
    }                                                           \

#define MATH_TESTS_FLOAT(func) MATH_TESTS_LIMITS(float, float, func, flt_err, 0.05f, 0.95f)
#define MATH_TESTS_DOUBLE(func) MATH_TESTS_LIMITS(double, double, func, dbl_err, 0.05, 0.95)

MATH_TESTS_FLOAT(sin)
MATH_TESTS_FLOAT(cos)
MATH_TESTS_FLOAT(tan)
MATH_TESTS_FLOAT(asin)
MATH_TESTS_FLOAT(acos)
MATH_TESTS_FLOAT(atan)

MATH_TESTS_FLOAT(sinh)
MATH_TESTS_FLOAT(cosh)
MATH_TESTS_FLOAT(tanh)


MATH_TESTS_FLOAT(sqrt)

MATH_TESTS_FLOAT(exp)
MATH_TESTS_FLOAT(log)
MATH_TESTS_FLOAT(log10)
MATH_TESTS_FLOAT(log2)

MATH_TESTS_LIMITS(float, float, abs, flt_err, -10, 10)
MATH_TESTS_LIMITS(float, float, ceil, flt_err, -10, 10)
MATH_TESTS_LIMITS(float, float, floor, flt_err, -10, 10)

MATH_TESTS_DOUBLE(sin)
MATH_TESTS_DOUBLE(cos)
MATH_TESTS_DOUBLE(tan)
MATH_TESTS_DOUBLE(asin)
MATH_TESTS_DOUBLE(acos)
MATH_TESTS_DOUBLE(atan)

MATH_TESTS_DOUBLE(sinh)
MATH_TESTS_DOUBLE(cosh)
MATH_TESTS_DOUBLE(tanh)
#if __cplusplus > 199711L
MATH_TESTS_FLOAT(asinh)
MATH_TESTS_FLOAT(atanh)
MATH_TESTS_LIMITS(float, float, acosh, flt_err, 1, 5)
MATH_TESTS_LIMITS(float, float, round, flt_err, -10, 10)
MATH_TESTS_FLOAT(cbrt)
MATH_TESTS_FLOAT(expm1)
MATH_TESTS_FLOAT(log1p)
MATH_TESTS_FLOAT(erf)
MATH_TESTS_FLOAT(erfc)

MATH_TESTS_DOUBLE(asinh)
MATH_TESTS_DOUBLE(atanh)
MATH_TESTS_LIMITS(double, double, acosh, dbl_err, 1, 5)
MATH_TESTS_LIMITS(double, double, round, dbl_err, -10, 10)
MATH_TESTS_DOUBLE(cbrt)
MATH_TESTS_DOUBLE(expm1)
MATH_TESTS_DOUBLE(erf)
MATH_TESTS_DOUBLE(log1p)
MATH_TESTS_DOUBLE(erfc)
#endif

MATH_TESTS_DOUBLE(sqrt)

MATH_TESTS_DOUBLE(exp)
MATH_TESTS_DOUBLE(log)
MATH_TESTS_DOUBLE(log10)
MATH_TESTS_DOUBLE(log2)

MATH_TESTS_LIMITS(double, double, abs, dbl_err, -10, 10)
MATH_TESTS_LIMITS(double, double, ceil, dbl_err, -10, 10)
MATH_TESTS_LIMITS(double, double, floor, dbl_err, -10, 10)
