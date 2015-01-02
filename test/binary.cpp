/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <af/array.h>
#include <af/arith.h>
#include <af/data.h>
#include <testHelpers.hpp>

using namespace std;
using namespace af;

const int num = 10000;

#define add(left, right) (left) + (right)
#define sub(left, right) (left) - (right)
#define mul(left, right) (left) * (right)
#define div(left, right) (left) / (right)

template<typename T> T mod(T a, T b)
{
    return std::fmod(a, b);
}

#define BINARY_TESTS(Ta, Tb, Tc, func)                                  \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb)                        \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randu(num, ta);                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = func(a, b);                                       \
        Ta *h_a = a.host<Ta>();                                         \
        Tb *h_b = b.host<Tb>();                                         \
        Tc *h_c = c.host<Tc>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], func(h_a[i], h_b[i])) <<                  \
                "for values: " << h_a[i]  << "," << h_b[i] << std::endl; \
        delete[] h_a;                                                   \
        delete[] h_b;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \
                                                                        \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_left)                 \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af::array a = randu(num, ta);                                   \
        Tb h_b = 3.0;                                                   \
        af::array c = func(a, h_b);                                     \
        Ta *h_a = a.host<Ta>();                                         \
        Ta *h_c = c.host<Ta>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], func(h_a[i], h_b)) <<                     \
                "for values: " << h_a[i]  << "," << h_b << std::endl;   \
        delete[] h_a;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \
                                                                        \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_right)                \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        Ta h_a = 5.0;                                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = func(h_a, b);                                     \
        Tb *h_b = b.host<Tb>();                                         \
        Tb *h_c = c.host<Tb>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], func(h_a, h_b[i])) <<                     \
                "for values: " << h_a  << "," << h_b[i] << std::endl;   \
        delete[] h_b;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \


#define BINARY_TESTS_NEAR(Ta, Tb, Tc, func, err)                        \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb)                        \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randu(num, ta);                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = func(a, b);                                       \
        Ta *h_a = a.host<Ta>();                                         \
        Tb *h_b = b.host<Tb>();                                         \
        Tc *h_c = c.host<Tc>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_NEAR(h_c[i], func(h_a[i], h_b[i]), err) <<           \
                "for values: " << h_a[i]  << "," << h_b[i] << std::endl; \
        delete[] h_a;                                                   \
        delete[] h_b;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \
                                                                        \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_left)                 \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af::array a = randu(num, ta);                                   \
        Tb h_b = 0.3;                                                   \
        af::array c = func(a, h_b);                                     \
        Ta *h_a = a.host<Ta>();                                         \
        Ta *h_c = c.host<Ta>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_NEAR(h_c[i], func(h_a[i], h_b), err) <<              \
                "for values: " << h_a[i]  << "," << h_b << std::endl;   \
        delete[] h_a;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \
                                                                        \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_right)                \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        Ta h_a = 0.3;                                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = func(h_a, b);                                     \
        Tb *h_b = b.host<Tb>();                                         \
        Tb *h_c = c.host<Tb>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_NEAR(h_c[i], func(h_a, h_b[i]), err) <<              \
                "for values: " << h_a  << "," << h_b[i] << std::endl;   \
        delete[] h_b;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \

#define BINARY_TESTS_FLOAT(func) BINARY_TESTS(float, float, float, func)
#define BINARY_TESTS_DOUBLE(func) BINARY_TESTS(double, double, double, func)
#define BINARY_TESTS_CFLOAT(func) BINARY_TESTS(cfloat, cfloat, cfloat, func)
#define BINARY_TESTS_CDOUBLE(func) BINARY_TESTS(cdouble, cdouble, cdouble, func)

#define BINARY_TESTS_INT(func) BINARY_TESTS(int, int, int, func)
#define BINARY_TESTS_UINT(func) BINARY_TESTS(uint, uint, uint, func)
#define BINARY_TESTS_NEAR_FLOAT(func) BINARY_TESTS_NEAR(float, float, float, func, 1e-5)
#define BINARY_TESTS_NEAR_DOUBLE(func) BINARY_TESTS_NEAR(double, double, double, func, 1e-10)

BINARY_TESTS_FLOAT(add)
BINARY_TESTS_FLOAT(sub)
BINARY_TESTS_FLOAT(mul)
BINARY_TESTS_NEAR(float, float, float, div, 1e-3) // FIXME
BINARY_TESTS_FLOAT(min)
BINARY_TESTS_FLOAT(max)
BINARY_TESTS_NEAR(float, float, float, mod, 1e-5) // FIXME

BINARY_TESTS_DOUBLE(add)
BINARY_TESTS_DOUBLE(sub)
BINARY_TESTS_DOUBLE(mul)
BINARY_TESTS_DOUBLE(div)
BINARY_TESTS_DOUBLE(min)
BINARY_TESTS_DOUBLE(max)
BINARY_TESTS_DOUBLE(mod)

BINARY_TESTS_NEAR_FLOAT(atan2)
BINARY_TESTS_NEAR_FLOAT(pow)
BINARY_TESTS_NEAR_FLOAT(hypot)

BINARY_TESTS_NEAR_DOUBLE(atan2)
BINARY_TESTS_NEAR_DOUBLE(pow)
BINARY_TESTS_NEAR_DOUBLE(hypot)

BINARY_TESTS_INT(add)
BINARY_TESTS_INT(sub)
BINARY_TESTS_INT(mul)

BINARY_TESTS_UINT(add)
BINARY_TESTS_UINT(sub)
BINARY_TESTS_UINT(mul)

BINARY_TESTS_CFLOAT(add)
BINARY_TESTS_CFLOAT(sub)

BINARY_TESTS_CDOUBLE(add)
BINARY_TESTS_CDOUBLE(sub)

// Mixed types
BINARY_TESTS_NEAR(float, double, double, add, 1e-5)
BINARY_TESTS_NEAR(float, double, double, sub, 1e-5)
BINARY_TESTS_NEAR(float, double, double, mul, 1e-5)
BINARY_TESTS_NEAR(float, double, double, div, 1e-5)
