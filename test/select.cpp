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

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::eval;
using af::NaN;
using af::randu;
using af::select;
using af::seq;
using af::span;
using af::sum;
using std::string;
using std::stringstream;
using std::vector;

template<typename T>
class Select : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, uint, int, intl, uintl,
                         uchar, char, short, ushort, half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Select, TestTypes);

template<typename T>
void selectTest(const dim4& dims) {
    SUPPORTED_TYPE_CHECK(T);
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
void selectScalarTest(const dim4& dims) {
    SUPPORTED_TYPE_CHECK(T);
    dtype ty = (dtype)dtype_traits<T>::af_type;

    array a    = randu(dims, ty);
    array cond = randu(dims, ty) > a;
    double b   = 3;

    if (a.isinteger()) { a = (a % (1 << 30)).as(ty); }

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

TYPED_TEST(Select, Simple) { selectTest<TypeParam>(dim4(1024, 1024)); }

TYPED_TEST(Select, RightScalar) {
    selectScalarTest<TypeParam, true>(dim4(1000, 1000));
}

TYPED_TEST(Select, LeftScalar) {
    selectScalarTest<TypeParam, true>(dim4(1000, 1000));
}

TEST(Select, NaN) {
    dim4 dims(1000, 1250);
    dtype ty = f32;

    array a                                 = randu(dims, ty);
    a(seq(a.dims(0) / 2), span, span, span) = NaN;
    float b                                 = 0;
    array c                                 = select(isNaN(a), b, a);

    int num = (int)a.elements();

    vector<float> ha(num);
    vector<float> hc(num);

    a.host(&ha[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(hc[i], std::isnan(ha[i]) ? b : ha[i]);
    }
}

TEST(Select, ISSUE_1249) {
    dim4 dims(2, 3, 4);
    array cond = randu(dims) > 0.5;
    array a    = randu(dims);
    array b    = select(cond, a - a * 0.9, a);
    array c    = a - a * cond * 0.9;

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        EXPECT_NEAR(hc[i], hb[i], 1e-7) << "at " << i;
    }
}

TEST(Select, 4D) {
    dim4 dims(2, 3, 4, 2);
    array cond = randu(dims) > 0.5;
    array a    = randu(dims);
    array b    = select(cond, a - a * 0.9, a);
    array c    = a - a * cond * 0.9;

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        EXPECT_NEAR(hc[i], hb[i], 1e-7) << "at " << i;
    }
}

TEST(Select, Issue_1730) {
    const int n = 1000;
    const int m = 200;
    array a     = randu(n, m) - 0.5;
    eval(a);

    vector<float> ha1(a.elements());
    a.host(&ha1[0]);

    const int n1 = n / 2;
    const int n2 = n1 + n / 4;

    a(seq(n1, n2), span) =
        select(a(seq(n1, n2), span) >= 0, a(seq(n1, n2), span),
               a(seq(n1, n2), span) * -1);

    vector<float> ha2(a.elements());
    a.host(&ha2[0]);

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            if (i < n1 || i > n2) {
                ASSERT_FLOAT_EQ(ha1[i], ha2[i])
                    << "at (" << i << ", " << j << ")";
            } else {
                ASSERT_FLOAT_EQ(ha2[i], (ha1[i] >= 0 ? ha1[i] : -ha1[i]))
                    << "at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST(Select, Issue_1730_scalar) {
    const int n = 1000;
    const int m = 200;
    array a     = randu(n, m) - 0.5;
    eval(a);

    vector<float> ha1(a.elements());
    a.host(&ha1[0]);

    const int n1 = n / 2;
    const int n2 = n1 + n / 4;

    float val = 0;
    a(seq(n1, n2), span) =
        select(a(seq(n1, n2), span) >= 0, a(seq(n1, n2), span), val);

    vector<float> ha2(a.elements());
    a.host(&ha2[0]);

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            if (i < n1 || i > n2) {
                ASSERT_FLOAT_EQ(ha1[i], ha2[i])
                    << "at (" << i << ", " << j << ")";
            } else {
                ASSERT_FLOAT_EQ(ha2[i], (ha1[i] >= 0 ? ha1[i] : val))
                    << "at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST(Select, MaxDim) {
    const size_t largeDim = 65535 * 32 + 1;

    array a    = constant(1, largeDim);
    array b    = constant(0, largeDim);
    array cond = constant(0, largeDim, b8);

    array sel = select(cond, a, b);
    float sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = constant(1, 1, largeDim);
    b    = constant(0, 1, largeDim);
    cond = constant(0, 1, largeDim, b8);

    sel = select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = constant(1, 1, 1, largeDim);
    b    = constant(0, 1, 1, largeDim);
    cond = constant(0, 1, 1, largeDim, b8);

    sel = select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);

    a    = constant(1, 1, 1, 1, largeDim);
    b    = constant(0, 1, 1, 1, largeDim);
    cond = constant(0, 1, 1, 1, largeDim, b8);

    sel = select(cond, a, b);
    sum = af::sum<float>(sel);

    ASSERT_FLOAT_EQ(sum, 0.f);
}

struct select_params {
    dim4 out;
    dim4 cond;
    dim4 a;
    dim4 b;
    select_params(dim4 out_, dim4 cond_, dim4 a_, dim4 b_)
        : out(out_), cond(cond_), a(a_), b(b_) {}
};

class Select_ : public ::testing::TestWithParam<select_params> {};

string pd4(dim4 dims) {
    string out(32, '\0');
    int len = snprintf(const_cast<char*>(out.data()), 32, "%lld_%lld_%lld_%lld",
                       dims[0], dims[1], dims[2], dims[3]);
    out.resize(len);
    return out;
}

string testNameGenerator(
    const ::testing::TestParamInfo<Select_::ParamType> info) {
    stringstream ss;
    ss << "out_" << pd4(info.param.out) << "_cond_" << pd4(info.param.cond)
       << "_a_" << pd4(info.param.a) << "_b_" << pd4(info.param.b);
    return ss.str();
}

vector<select_params> getSelectTestParams(int M, int N) {
    const select_params _[] = {
        select_params(dim4(M), dim4(M), dim4(M), dim4(M)),
        select_params(dim4(M, N), dim4(M, N), dim4(M, N), dim4(M, N)),
        select_params(dim4(M, N, N), dim4(M, N, N), dim4(M, N, N),
                      dim4(M, N, N)),
        select_params(dim4(M, N, N, N), dim4(M, N, N, N), dim4(M, N, N, N),
                      dim4(M, N, N, N)),
        select_params(dim4(M, N), dim4(M, 1), dim4(M, 1), dim4(M, N)),
        select_params(dim4(M, N), dim4(M, 1), dim4(M, N), dim4(M, 1)),
        select_params(dim4(M, N), dim4(M, 1), dim4(M, N), dim4(M, N)),
        select_params(dim4(M, N), dim4(M, N), dim4(M, 1), dim4(M, N)),
        select_params(dim4(M, N), dim4(M, N), dim4(M, N), dim4(M, 1)),
        select_params(dim4(M, N), dim4(M, N), dim4(M, 1), dim4(M, 1))};
    return vector<select_params>(_, _ + sizeof(_) / sizeof(_[0]));
}

INSTANTIATE_TEST_SUITE_P(SmallDims, Select_,
                         ::testing::ValuesIn(getSelectTestParams(10, 5)),
                         testNameGenerator);

INSTANTIATE_TEST_SUITE_P(Dims33_9, Select_,
                         ::testing::ValuesIn(getSelectTestParams(33, 9)),
                         testNameGenerator);

INSTANTIATE_TEST_SUITE_P(DimsLg, Select_,
                         ::testing::ValuesIn(getSelectTestParams(512, 32)),
                         testNameGenerator);

TEST_P(Select_, Batch) {
    select_params params = GetParam();

    float aval = 5.0f;
    float bval = 10.0f;
    array a    = constant(aval, params.a);
    array b    = constant(bval, params.b);
    array cond = (iota(params.cond) % 2).as(b8);

    array out = select(cond, a, b);

    EXPECT_EQ(out.dims(), params.out);

    vector<float> h_out(out.elements());
    out.host(h_out.data());
    vector<unsigned char> h_cond(cond.elements());
    cond.host(h_cond.data());

    vector<float> gold(params.out.elements());
    for (size_t i = 0; i < gold.size(); i++) {
        gold[i] = h_cond[i % h_cond.size()] ? aval : bval;
        ASSERT_FLOAT_EQ(gold[i], h_out[i]) << "at: " << i;
    }
}

struct selectlr_params {
    dim4 out;
    dim4 cond;
    dim4 ab;
    selectlr_params(dim4 out_, dim4 cond_, dim4 ab_)
        : out(out_), cond(cond_), ab(ab_) {}
};

class SelectLR_ : public ::testing::TestWithParam<selectlr_params> {};

vector<selectlr_params> getSelectLRTestParams(int M, int N) {
    const selectlr_params _[] = {
        selectlr_params(dim4(M), dim4(M), dim4(M)),
        selectlr_params(dim4(M, N), dim4(M, N), dim4(M, N)),
        selectlr_params(dim4(M, N, N), dim4(M, N, N), dim4(M, N, N)),
        selectlr_params(dim4(M, N, N, N), dim4(M, N, N, N), dim4(M, N, N, N)),
        selectlr_params(dim4(M, N), dim4(M, 1), dim4(M, N)),
        selectlr_params(dim4(M, N), dim4(M, N), dim4(M, 1))};

    return vector<selectlr_params>(_, _ + sizeof(_) / sizeof(_[0]));
}

string testNameGeneratorLR(
    const ::testing::TestParamInfo<SelectLR_::ParamType> info) {
    stringstream ss;
    ss << "out_" << pd4(info.param.out) << "_cond_" << pd4(info.param.cond)
       << "_ab_" << pd4(info.param.ab);
    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(SmallDims, SelectLR_,
                         ::testing::ValuesIn(getSelectLRTestParams(10, 5)),
                         testNameGeneratorLR);

INSTANTIATE_TEST_SUITE_P(Dims33_9, SelectLR_,
                         ::testing::ValuesIn(getSelectLRTestParams(33, 9)),
                         testNameGeneratorLR);

INSTANTIATE_TEST_SUITE_P(DimsLg, SelectLR_,
                         ::testing::ValuesIn(getSelectLRTestParams(512, 32)),
                         testNameGeneratorLR);

TEST_P(SelectLR_, BatchL) {
    selectlr_params params = GetParam();

    float aval = 5.0f;
    float bval = 10.0f;
    array b    = constant(bval, params.ab);
    array cond = (iota(params.cond) % 2).as(b8);

    array out = select(cond, static_cast<double>(aval), b);

    EXPECT_EQ(out.dims(), params.out);

    vector<float> h_out(out.elements());
    out.host(h_out.data());
    vector<unsigned char> h_cond(cond.elements());
    cond.host(h_cond.data());

    vector<float> gold(params.out.elements());
    for (size_t i = 0; i < gold.size(); i++) {
        gold[i] = h_cond[i % h_cond.size()] ? aval : bval;
        ASSERT_FLOAT_EQ(gold[i], h_out[i]) << "at: " << i;
    }
}

TEST_P(SelectLR_, BatchR) {
    selectlr_params params = GetParam();

    float aval = 5.0f;
    float bval = 10.0f;
    array a    = constant(aval, params.ab);
    array cond = (iota(params.cond) % 2).as(b8);

    array out = select(cond, a, static_cast<double>(bval));

    EXPECT_EQ(out.dims(), params.out);

    vector<float> h_out(out.elements());
    out.host(h_out.data());
    vector<unsigned char> h_cond(cond.elements());
    cond.host(h_cond.data());

    vector<float> gold(params.out.elements());
    for (size_t i = 0; i < gold.size(); i++) {
        gold[i] = h_cond[i % h_cond.size()] ? aval : bval;
        ASSERT_FLOAT_EQ(gold[i], h_out[i]) << "at: " << i;
    }
}

TEST(Select, InvalidSizeOfAB) {
    af_array a    = 0;
    af_array b    = 0;
    af_array cond = 0;
    af_array out  = 0;

    double val = 0;
    dim_t dims = 10;
    ASSERT_SUCCESS(af_constant(&a, val, 1, &dims, f32));

    dims = 9;
    ASSERT_SUCCESS(af_constant(&b, val, 1, &dims, f32));

    dims = 10;
    ASSERT_SUCCESS(af_constant(&cond, val, 1, &dims, b8));

    ASSERT_EQ(AF_ERR_SIZE, af_select(&out, cond, a, b));

    char* msg = NULL;
    dim_t len = 0;
    af_get_last_error(&msg, &len);
    af_free_host(msg);
    af_release_array(a);
    af_release_array(b);
    af_release_array(cond);
}

TEST(Select, InvalidSizeOfCond) {
    af_array a    = 0;
    af_array b    = 0;
    af_array cond = 0;
    af_array out  = 0;

    double val = 0;
    dim_t dims = 10;
    ASSERT_SUCCESS(af_constant(&a, val, 1, &dims, f32));

    dims = 10;
    ASSERT_SUCCESS(af_constant(&b, val, 1, &dims, f32));

    dims = 9;
    ASSERT_SUCCESS(af_constant(&cond, val, 1, &dims, b8));

    ASSERT_EQ(AF_ERR_SIZE, af_select(&out, cond, a, b));

    char* msg = NULL;
    dim_t len = 0;
    af_get_last_error(&msg, &len);
    af_free_host(msg);
    af_release_array(a);
    af_release_array(b);
    af_release_array(cond);
}

TEST(Select, SNIPPET_select) {
    //! [ex_data_select]
    int elements = 9;
    char hCond[] = {1, 0, 1, 0, 1, 0, 1, 0, 1};
    float hA[]   = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    float hB[]   = {3, 3, 3, 3, 3, 3, 3, 3, 3};

    array cond(elements, hCond);
    array a(elements, hA);
    array b(elements, hB);

    array out = select(cond, a, b);
    // out = {2, 3, 2, 3, 2, 3, 2, 3, 2};
    //! [ex_data_select]

    //! [ex_data_select_c]
    vector<float> hOut(elements);
    for (size_t i = 0; i < hOut.size(); i++) {
        if (hCond[i]) {
            hOut[i] = hA[i];
        } else {
            hOut[i] = hB[i];
        }
    }
    //! [ex_data_select_c]

    ASSERT_VEC_ARRAY_EQ(hOut, dim4(9), out);
}
