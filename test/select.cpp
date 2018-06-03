/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using namespace af;

template<typename T>
class Select : public ::testing::Test
{
};

typedef ::testing::Types<float, double, af::cfloat, af::cdouble, uint, int, intl, uintl, uchar, char, short, ushort> TestTypes;
TYPED_TEST_CASE(Select, TestTypes);

template<typename T>
void selectTest(const dim4 &dims)
{
    if (noDoubleTests<T>()) return;
    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    array a = randu(dims, ty);
    array b = randu(dims, ty);

    if (a.isinteger()) {
        a = (a % (1 << 30)).as(ty);
        b = (b % (1 << 30)).as(ty);
    }

    array cond = randu(dims, ty) > a;

    array c = select(cond, a, b);

    int num = (int)a.elements();

    std::vector<T> ha(num);
    std::vector<T> hb(num);
    std::vector<T> hc(num);
    std::vector<char> hcond(num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    c.host(&hc[0]);
    cond.host(&hcond[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hc[i], hcond[i] ? ha[i] : hb[i]);
    }
}

template<typename T, bool is_right>
void selectScalarTest(const dim4 &dims)
{
    if (noDoubleTests<T>()) return;
    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    array a = randu(dims, ty);
    array cond = randu(dims, ty) > a;
    double b = 3;

    if (a.isinteger()) {
        a = (a % (1 << 30)).as(ty);
    }

    array c = is_right ? select(cond, a, b) : select(cond, b, a);

    int num = (int)a.elements();

    std::vector<T> ha(num);
    std::vector<T> hc(num);
    std::vector<char> hcond(num);

    a.host(&ha[0]);
    c.host(&hc[0]);
    cond.host(&hcond[0]);

    if (is_right) {
        for (int i = 0; i < num; i++) {
            ASSERT_EQ(hc[i], hcond[i] ? ha[i] : T(b));
        }
    } else {
        for (int i = 0; i < num; i++) {
            ASSERT_EQ(hc[i], hcond[i] ? T(b) : ha[i]);
        }
    }
}

TYPED_TEST(Select, Simple)
{
    selectTest<TypeParam>(dim4(1024, 1024));
}

TYPED_TEST(Select, RightScalar)
{
    selectScalarTest<TypeParam, true>(dim4(1000, 1000));
}

TYPED_TEST(Select, LeftScalar)
{
    selectScalarTest<TypeParam, true>(dim4(1000, 1000));
}

TEST(Select, NaN)
{
    dim4 dims(1000, 1250);
    af::dtype ty = f32;

    array a = randu(dims, ty);
    a(seq(a.dims(0) / 2), span, span, span) = af::NaN;
    float b = 0;
    array c = select(isNaN(a), b, a);

    int num = (int)a.elements();

    std::vector<float> ha(num);
    std::vector<float> hc(num);

    a.host(&ha[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(hc[i], std::isnan(ha[i]) ? b : ha[i]);
    }
}

TEST(Select, ISSUE_1249)
{
    dim4 dims(2, 3, 4);
    array cond = af::randu(dims) > 0.5;
    array a = af::randu(dims);
    array b = select(cond, a - a * 0.9, a);
    array c = a - a * cond * 0.9;

    int num = (int)dims.elements();
    std::vector<float> hb(num);
    std::vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        EXPECT_NEAR(hc[i], hb[i], 1e-7) << "at " << i;
    }
}

TEST(Select, 4D)
{
    dim4 dims(2, 3, 4, 2);
    array cond = af::randu(dims) > 0.5;
    array a = af::randu(dims);
    array b = select(cond, a - a * 0.9, a);
    array c = a - a * cond * 0.9;

    int num = (int)dims.elements();
    std::vector<float> hb(num);
    std::vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        EXPECT_NEAR(hc[i], hb[i], 1e-7) << "at " << i;
    }
}

TEST(Select, Issue_1730)
{
    const int n = 1000;
    const int m = 200;
    af::array a = af::randu(n, m) - 0.5;
    af::eval(a);

    std::vector<float> ha1(a.elements());
    a.host(&ha1[0]);

    const int n1 = n / 2;
    const int n2 = n1 + n / 4;

    a(af::seq(n1, n2), af::span) =
        af::select(a(af::seq(n1, n2), af::span) >= 0,
                   a(af::seq(n1, n2), af::span),
                   a(af::seq(n1, n2), af::span) * -1);

    std::vector<float> ha2(a.elements());
    a.host(&ha2[0]);

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            if (i < n1 || i > n2) {
                ASSERT_FLOAT_EQ(ha1[i], ha2[i]) << "at (" << i << ", " << j << ")";
            } else {
                ASSERT_FLOAT_EQ(ha2[i], (ha1[i] >= 0 ? ha1[i] : -ha1[i]))  << "at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST(Select, Issue_1730_scalar)
{
    const int n = 1000;
    const int m = 200;
    af::array a = af::randu(n, m) - 0.5;
    af::eval(a);

    std::vector<float> ha1(a.elements());
    a.host(&ha1[0]);

    const int n1 = n / 2;
    const int n2 = n1 + n / 4;

    float val = 0;
    a(af::seq(n1, n2), af::span) =
        af::select(a(af::seq(n1, n2), af::span) >= 0,
                   a(af::seq(n1, n2), af::span),
                   val);

    std::vector<float> ha2(a.elements());
    a.host(&ha2[0]);

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            if (i < n1 || i > n2) {
                ASSERT_FLOAT_EQ(ha1[i], ha2[i]) << "at (" << i << ", " << j << ")";
            } else {
                ASSERT_FLOAT_EQ(ha2[i], (ha1[i] >= 0 ? ha1[i] : val))  << "at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST(Select, MaxDim)
{
    const size_t largeDim = 65535 * 32 + 1;

    af::array a    = af::constant(1, largeDim);
    af::array b    = af::constant(0, largeDim);
    af::array cond = af::constant(0, largeDim, b8);

    af::array sel  = af::select(cond, a, b);
    float sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = af::constant(1, 1, largeDim);
    b    = af::constant(0, 1, largeDim);
    cond = af::constant(0, 1, largeDim, b8);

    sel  = af::select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = af::constant(1, 1, 1, largeDim);
    b    = af::constant(0, 1, 1, largeDim);
    cond = af::constant(0, 1, 1, largeDim, b8);

    sel  = af::select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = af::constant(1, 1, 1, 1, largeDim);
    b    = af::constant(0, 1, 1, 1, largeDim);
    cond = af::constant(0, 1, 1, 1, largeDim, b8);

    sel  = af::select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);
}
