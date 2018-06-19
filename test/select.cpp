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
using af::NaN;
using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::eval;
using af::randu;
using af::select;
using af::seq;
using af::span;
using af::sum;


template<typename T>
class Select : public ::testing::Test
{
};

typedef ::testing::Types<float, double, cfloat, cdouble, uint, int, intl, uintl, uchar, char, short, ushort> TestTypes;
TYPED_TEST_CASE(Select, TestTypes);

template<typename T>
void selectTest(const dim4 &dims)
{
    if (noDoubleTests<T>()) return;
    dtype ty = (dtype)dtype_traits<T>::af_type;

    array a = randu(dims, ty);
    array b = randu(dims, ty);

    if (a.isinteger()) {
        a = (a % (1 << 30)).as(ty);
        b = (b % (1 << 30)).as(ty);
    }

    array cond = randu(dims, ty) > a;

    array c = select(cond, a, b);

    int num = (int)a.elements();

    vector<T> ha(num);
    vector<T> hb(num);
    vector<T> hc(num);
    vector<char> hcond(num);

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
    dtype ty = (dtype)dtype_traits<T>::af_type;

    array a = randu(dims, ty);
    array cond = randu(dims, ty) > a;
    double b = 3;

    if (a.isinteger()) {
        a = (a % (1 << 30)).as(ty);
    }

    array c = is_right ? select(cond, a, b) : select(cond, b, a);

    int num = (int)a.elements();

    vector<T> ha(num);
    vector<T> hc(num);
    vector<char> hcond(num);

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
    dtype ty = f32;

    array a = randu(dims, ty);
    a(seq(a.dims(0) / 2), span, span, span) = NaN;
    float b = 0;
    array c = select(isNaN(a), b, a);

    int num = (int)a.elements();

    vector<float> ha(num);
    vector<float> hc(num);

    a.host(&ha[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(hc[i], std::isnan(ha[i]) ? b : ha[i]);
    }
}

TEST(Select, ISSUE_1249)
{
    dim4 dims(2, 3, 4);
    array cond = randu(dims) > 0.5;
    array a = randu(dims);
    array b = select(cond, a - a * 0.9, a);
    array c = a - a * cond * 0.9;

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        EXPECT_NEAR(hc[i], hb[i], 1e-7) << "at " << i;
    }
}

TEST(Select, 4D)
{
    dim4 dims(2, 3, 4, 2);
    array cond = randu(dims) > 0.5;
    array a = randu(dims);
    array b = select(cond, a - a * 0.9, a);
    array c = a - a * cond * 0.9;

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

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
    array a = randu(n, m) - 0.5;
    eval(a);

    vector<float> ha1(a.elements());
    a.host(&ha1[0]);

    const int n1 = n / 2;
    const int n2 = n1 + n / 4;

    a(seq(n1, n2), span) =
        select(a(seq(n1, n2), span) >= 0,
                   a(seq(n1, n2), span),
                   a(seq(n1, n2), span) * -1);

    vector<float> ha2(a.elements());
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
    array a = randu(n, m) - 0.5;
    eval(a);

    vector<float> ha1(a.elements());
    a.host(&ha1[0]);

    const int n1 = n / 2;
    const int n2 = n1 + n / 4;

    float val = 0;
    a(seq(n1, n2), span) =
        select(a(seq(n1, n2), span) >= 0,
                   a(seq(n1, n2), span),
                   val);

    vector<float> ha2(a.elements());
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

    array a    = constant(1, largeDim);
    array b    = constant(0, largeDim);
    array cond = constant(0, largeDim, b8);

    array sel  = select(cond, a, b);
    float sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = constant(1, 1, largeDim);
    b    = constant(0, 1, largeDim);
    cond = constant(0, 1, largeDim, b8);

    sel  = select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = constant(1, 1, 1, largeDim);
    b    = constant(0, 1, 1, largeDim);
    cond = constant(0, 1, 1, largeDim, b8);

    sel  = select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = constant(1, 1, 1, 1, largeDim);
    b    = constant(0, 1, 1, 1, largeDim);
    cond = constant(0, 1, 1, 1, largeDim, b8);

    sel  = select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);
}
