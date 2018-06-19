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

typedef std::complex<float> complex_float;
typedef std::complex<double> complex_double;

template<typename T> T mod(T a, T b)
{
    return std::fmod(a, b);
}

af::array randgen(const int num, dtype ty)
{
    af::array tmp = round(1 + 2 * af::randu(num, f32)).as(ty);
    tmp.eval();
    return tmp;
}

#define MY_ASSERT_NEAR(aa, bb, cc) ASSERT_NEAR(abs(aa), abs(bb), (cc))

#define BINARY_TESTS(Ta, Tb, Tc, func)                                  \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb)                        \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randgen(num, ta);                                 \
        af::array b = randgen(num, tb);                                 \
        af::array c = func(a, b);                                       \
        Ta *h_a = a.host<Ta>();                                         \
        Tb *h_b = b.host<Tb>();                                         \
        Tc *h_c = c.host<Tc>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], func(h_a[i], h_b[i])) <<                  \
                "for values: " << h_a[i]  << "," << h_b[i] << endl;     \
        af_free_host(h_a);                                              \
        af_free_host(h_b);                                              \
        af_free_host(h_c);                                              \
    }                                                                   \
                                                                        \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_left)                 \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af::array a = randgen(num, ta);                                 \
        Tb h_b = 3.0;                                                   \
        af::array c = func(a, h_b);                                     \
        Ta *h_a = a.host<Ta>();                                         \
        Ta *h_c = c.host<Ta>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], func(h_a[i], h_b)) <<                     \
                "for values: " << h_a[i]  << "," << h_b << endl;   \
        af_free_host(h_a);                                              \
        af_free_host(h_c);                                              \
    }                                                                   \
                                                                        \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_right)                \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        Ta h_a = 5.0;                                                   \
        af::array b = randgen(num, tb);                                 \
        af::array c = func(h_a, b);                                     \
        Tb *h_b = b.host<Tb>();                                         \
        Tb *h_c = c.host<Tb>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], func(h_a, h_b[i])) <<                     \
                "for values: " << h_a  << "," << h_b[i] << endl;        \
        af_free_host(h_b);                                              \
        af_free_host(h_c);                                              \
    }                                                                   \


#define BINARY_TESTS_NEAR_GENERAL(Ta, Tb, Tc, Td, Te,func, err)         \
    TEST(BinaryTestsFloating, Test_##func##_##Ta##_##Tb)                \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randgen(num, ta);                                 \
        af::array b = randgen(num, tb);                                 \
        af::array c = func(a, b);                                       \
        Ta *h_a = a.host<Ta>();                                         \
        Tb *h_b = b.host<Tb>();                                         \
        Tc *h_c = c.host<Tc>();                                         \
        for (int i = 0; i < num; i++)                                   \
            MY_ASSERT_NEAR(h_c[i], func(h_a[i], h_b[i]), (err)) <<      \
                "for values: " << h_a[i]  << "," << h_b[i] << endl;     \
        af_free_host(h_a);                                              \
        af_free_host(h_b);                                              \
        af_free_host(h_c);                                              \
    }                                                                   \
                                                                        \
    TEST(BinaryTestsFloating, Test_##func##_##Ta##_##Tb##_left)         \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af::array a = randgen(num, ta);                                 \
        Tb h_b = 0.3;                                                   \
        af::array c = func(a, h_b);                                     \
        Ta *h_a = a.host<Ta>();                                         \
        Td *h_d = c.host<Td>();                                         \
        for (int i = 0; i < num; i++)                                   \
            MY_ASSERT_NEAR(h_d[i], func(h_a[i], h_b), err) <<           \
                "for values: " << h_a[i]  << "," << h_b << endl;        \
        af_free_host(h_a);                                              \
        af_free_host(h_d);                                              \
    }                                                                   \
                                                                        \
    TEST(BinaryTestsFloating, Test_##func##_##Ta##_##Tb##_right)        \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        Ta h_a = 0.3;                                                   \
        af::array b = randgen(num, tb);                                 \
        af::array c = func(h_a, b);                                     \
        Tb *h_b = b.host<Tb>();                                         \
        Te *h_e = c.host<Te>();                                         \
        for (int i = 0; i < num; i++)                                   \
            MY_ASSERT_NEAR(h_e[i], func(h_a, h_b[i]), err) <<           \
                "for values: " << h_a  << "," << h_b[i] << endl;        \
        af_free_host(h_b);                                              \
        af_free_host(h_e);                                              \
    }                                                                   \

#define BINARY_TESTS_NEAR(Ta, Tb, Tc, func, err) BINARY_TESTS_NEAR_GENERAL(Ta, Tb, Tc, Ta, Tc, func, err)

#define BINARY_TESTS_FLOAT(func) BINARY_TESTS(float, float, float, func)
#define BINARY_TESTS_DOUBLE(func) BINARY_TESTS(double, double, double, func)
#define BINARY_TESTS_CFLOAT(func) BINARY_TESTS(cfloat, cfloat, cfloat, func)
#define BINARY_TESTS_CDOUBLE(func) BINARY_TESTS(cdouble, cdouble, cdouble, func)

#define BINARY_TESTS_INT(func) BINARY_TESTS(int, int, int, func)
#define BINARY_TESTS_UINT(func) BINARY_TESTS(uint, uint, uint, func)
#define BINARY_TESTS_INTL(func) BINARY_TESTS(intl, intl, intl, func)
#define BINARY_TESTS_UINTL(func) BINARY_TESTS(uintl, uintl, uintl, func)
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

BINARY_TESTS_INTL(add)
BINARY_TESTS_INTL(sub)
BINARY_TESTS_INTL(mul)

BINARY_TESTS_UINTL(add)
BINARY_TESTS_UINTL(sub)
BINARY_TESTS_UINTL(mul)

BINARY_TESTS_CFLOAT(add)
BINARY_TESTS_CFLOAT(sub)

BINARY_TESTS_CDOUBLE(add)
BINARY_TESTS_CDOUBLE(sub)

// Mixed types
BINARY_TESTS_NEAR(float, double, double, add, 1e-5)
BINARY_TESTS_NEAR(float, double, double, sub, 1e-5)
BINARY_TESTS_NEAR(float, double, double, mul, 1e-5)
BINARY_TESTS_NEAR(float, double, double, div, 1e-5)

BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, add, 1e-5)
BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, sub, 1e-5)
BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, mul, 1e-5)
BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, div, 1e-5)

BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, add, 1e-5)
BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, sub, 1e-5)
BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, mul, 1e-5)
BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, div, 1e-5)

BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, add, 1e-5)
BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, sub, 1e-5)
BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, mul, 1e-5)
BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, div, 1e-5)

BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, add, 1e-5)
BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, sub, 1e-5)
BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, mul, 1e-5)
BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, div, 1e-5)


#define BITOP(func, T, op)                                  \
    TEST(BinaryTests, Test_##func##_##T)                    \
    {                                                       \
        af_dtype ty = (af_dtype)dtype_traits<T>::af_type;   \
        const T vala = 4095;                                \
        const T valb = 3;                                   \
        const T valc = vala op valb;                        \
        const int num = 10;                                 \
        af::array a = af::constant(vala, num, ty);          \
        af::array b = af::constant(valb, num, ty);          \
        af::array c = a op b;                               \
        T *h_a = a.host<T>();                               \
        T *h_b = b.host<T>();                               \
        T *h_c = c.host<T>();                               \
        for (int i = 0; i < num; i++)                       \
            ASSERT_EQ(h_c[i], valc) <<                      \
                "for values: " << h_a[i]  <<                \
                "," << h_b[i] << endl;                      \
        af_free_host(h_a);                                  \
        af_free_host(h_b);                                  \
        af_free_host(h_c);                                  \
    }                                                       \

BITOP(bitor, int, |)
BITOP(bitand, int, &)
BITOP(bitxor, int, ^)
BITOP(bitshiftl, int, <<)
BITOP(bitshiftr, int, >>)
BITOP(bitor, uint, |)
BITOP(bitand, uint, &)
BITOP(bitxor, uint, ^)
BITOP(bitshiftl, uint, <<)
BITOP(bitshiftr, uint, >>)

BITOP(bitor, intl, |)
BITOP(bitand, intl, &)
BITOP(bitxor, intl, ^)
BITOP(bitshiftl, intl, <<)
BITOP(bitshiftr, intl, >>)
BITOP(bitor, uintl, |)
BITOP(bitand, uintl, &)
BITOP(bitxor, uintl, ^)
BITOP(bitshiftl, uintl, <<)
BITOP(bitshiftr, uintl, >>)

TEST(BinaryTests, Test_pow_cfloat_float)
{
    af::array a = randgen(num, c32);
    af::array b = randgen(num, f32);
    af::array c = af::pow(a, b);
    complex_float *h_a = (complex_float *)a.host<cfloat>();
    float *h_b = b.host<float>();
    complex_float *h_c = (complex_float *)c.host<cfloat>();
    for (int i = 0; i < num; i++) {
        complex_float res = std::pow(h_a[i], h_b[i]);
        ASSERT_NEAR(real(h_c[i]), real(res), 1E-5)
            << "for real values of: " << h_a[i]  << "," << h_b[i] << endl;
        ASSERT_NEAR(imag(h_c[i]), imag(res), 1E-5)
            << "for imag values of: " << h_a[i]  << "," << h_b[i] << endl;

    }
    af_free_host(h_a);
    af_free_host(h_b);
    af_free_host(h_c);
}

TEST(BinaryTests, Test_pow_cdouble_cdouble)
{
    if (noDoubleTests<cdouble>()) return;
    af::array a = randgen(num, c64);
    af::array b = randgen(num, c64);
    af::array c = af::pow(a, b);
    complex_double *h_a = (complex_double *)a.host<cdouble>();
    complex_double *h_b = (complex_double *)b.host<cdouble>();
    complex_double *h_c = (complex_double *)c.host<cdouble>();
    for (int i = 0; i < num; i++) {
        complex_double res = std::pow(h_a[i], h_b[i]);
        ASSERT_NEAR(real(h_c[i]), real(res), 1E-10)
            << "for real values of: " << h_a[i]  << "," << h_b[i] << endl;
        ASSERT_NEAR(imag(h_c[i]), imag(res), 1E-10)
            << "for imag values of: " << h_a[i]  << "," << h_b[i] << endl;

    }
    af_free_host(h_a);
    af_free_host(h_b);
    af_free_host(h_c);
}

TEST(BinaryTests, ISSUE_1762)
{
    af::array zero = af::constant(0, 5, f32);
    af::array result = af::pow(zero, 2);
    vector<complex_float> hres(result.elements());
    result.host(&hres[0]);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(real(hres[i]), 0);
        ASSERT_EQ(imag(hres[i]), 0);
    }
}
