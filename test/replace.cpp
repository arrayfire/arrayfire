/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <half.hpp>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::NaN;
using af::randu;
using af::seq;
using af::span;
using std::vector;

template<typename T>
class Replace : public ::testing::Test {};

typedef ::testing::Types<half_float::half, float, double, cfloat, cdouble, uint,
                         int, intl, uintl, uchar, char, short, ushort>
    TestTypes;

TYPED_TEST_SUITE(Replace, TestTypes);

template<typename T>
void replaceTest(const dim4 &dims) {
    SUPPORTED_TYPE_CHECK(T);
    dtype ty = (dtype)dtype_traits<T>::af_type;

    array a = randu(dims, ty);
    array b = randu(dims, ty);

    if (a.isinteger()) {
        a = (a % (1 << 30)).as(ty);
        b = (b % (1 << 30)).as(ty);
    }

    array c = a.copy();

    array cond = randu(dims, ty) > a;

    replace(c, cond, b);

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

template<typename T>
void replaceScalarTest(const dim4 &dims) {
    SUPPORTED_TYPE_CHECK(T);
    dtype ty = (dtype)dtype_traits<T>::af_type;

    array a = randu(dims, ty);

    if (a.isinteger()) { a = (a % (1 << 30)).as(ty); }

    array c    = a.copy();
    array cond = randu(dims, ty) > a;
    double b   = 3;

    replace(c, cond, b);
    int num = (int)a.elements();

    vector<T> ha(num);
    vector<T> hc(num);
    vector<char> hcond(num);

    a.host(&ha[0]);
    c.host(&hc[0]);
    cond.host(&hcond[0]);

    for (int i = 0; i < num; i++) { ASSERT_EQ(hc[i], hcond[i] ? ha[i] : T(b)); }
}

TYPED_TEST(Replace, Simple) { replaceTest<TypeParam>(dim4(1024, 1024)); }

TYPED_TEST(Replace, Scalar) { replaceScalarTest<TypeParam>(dim4(5, 5)); }

TEST(Replace, NaN) {
    dim4 dims(1000, 1250);
    dtype ty = f32;

    array a                                 = randu(dims, ty);
    a(seq(a.dims(0) / 2), span, span, span) = NaN;
    array c                                 = a.copy();
    float b                                 = 0;
    replace(c, !isNaN(c), b);

    int num = (int)a.elements();

    vector<float> ha(num);
    vector<float> hc(num);

    a.host(&ha[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hc[i], (std::isnan(ha[i]) ? b : ha[i]));
    }
}

TEST(Replace, ISSUE_1249) {
    dim4 dims(2, 3, 4);
    array cond = randu(dims) > 0.5;
    array a    = randu(dims);
    array b    = a.copy();
    replace(b, !cond, a - a * 0.9);
    array c  = (a - a * 0.9);
    c(!cond) = a(!cond);

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(hc[i], hb[i]) << "at " << i;
    }
}

TEST(Replace, 4D) {
    dim4 dims(2, 3, 4, 2);
    array cond = randu(dims) > 0.5;
    array a    = randu(dims);
    array b    = a.copy();
    replace(b, !cond, a - a * 0.9);
    array c = a - a * cond * 0.9;

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(hc[i], hb[i]) << "at " << i;
    }
}

TEST(Replace, ISSUE_1683) {
    array A = randu(10, 20, f32);
    vector<float> ha1(A.elements());
    A.host(ha1.data());

    array B = A(0, span);
    replace(B, A(0, span) > 0.5, 0);

    vector<float> ha2(A.elements());
    A.host(ha2.data());

    vector<float> hb(B.elements());
    B.host(hb.data());

    // Ensures A is not modified by replace
    for (int i = 0; i < (int)A.elements(); i++) {
        ASSERT_FLOAT_EQ(ha1[i], ha2[i]);
    }

    // Ensures replace on B works as expected
    for (int i = 0; i < (int)B.elements(); i++) {
        float val = ha1[i * A.dims(0)];
        val       = val < 0.5 ? 0 : val;
        ASSERT_FLOAT_EQ(val, hb[i]);
    }
}
