/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/random.h>
#include <af/traits.hpp>

#include <sstream>
#include <string>
#include <vector>

using af::array;
using af::dim4;
using af::dtype;
using af::randu;
using std::abs;
using std::string;
using std::stringstream;
using std::vector;

const int num = 10000;

struct clamp_params {
    dim4 size_;
    dtype in_type_;
    dtype lo_type_;
    dtype hi_type_;
    dtype out_type_;

    clamp_params(dim4 size, dtype itype, dtype ltype, dtype htype, dtype otype)
        : size_(size)
        , in_type_(itype)
        , lo_type_(ltype)
        , hi_type_(htype)
        , out_type_(otype) {}
};

template<typename T>
class Clamp : public ::testing::TestWithParam<clamp_params> {
   public:
    void SetUp() {
        clamp_params params = GetParam();
        SUPPORTED_TYPE_CHECK(double);
        if (noDoubleTests(params.in_type_))
            GTEST_SKIP() << "Double not supported on this device";
        if (noHalfTests(params.in_type_))
            GTEST_SKIP() << "Half not supported on this device";
        if (noDoubleTests(params.hi_type_))
            GTEST_SKIP() << "Double not supported on this device";
        if (noHalfTests(params.hi_type_))
            GTEST_SKIP() << "Half not supported on this device";
        if (noDoubleTests(params.lo_type_))
            GTEST_SKIP() << "Double not supported on this device";
        if (noHalfTests(params.lo_type_))
            GTEST_SKIP() << "Half not supported on this device";

        in_ = randu(params.size_, params.in_type_);
        lo_ = randu(params.size_, params.lo_type_) / T(10);
        hi_ = T(1) - randu(params.size_, params.hi_type_) / T(10);
        lo_ = lo_.as(params.lo_type_);
        hi_ = hi_.as(params.hi_type_);

        size_t num = params.size_.elements();
        vector<T> hgold(num), hin(num), hlo(num), hhi(num);
        in_.as((dtype)af::dtype_traits<T>::af_type).host(&hin[0]);
        lo_.as((dtype)af::dtype_traits<T>::af_type).host(&hlo[0]);
        hi_.as((dtype)af::dtype_traits<T>::af_type).host(&hhi[0]);

        for (size_t i = 0; i < num; i++) {
            if (hin[i] < hlo[i])
                hgold[i] = hlo[i];
            else if (hin[i] > hhi[i])
                hgold[i] = hhi[i];
            else
                hgold[i] = hin[i];
        }

        gold_ = array(params.size_, &hgold[0]);
        gold_ = gold_.as(params.out_type_);
        gold_.eval();
    }

    af::array in_;
    af::array lo_;
    af::array hi_;
    af::array gold_;
};

string pd4(dim4 dims) {
    string out(32, '\0');
    int len = snprintf(const_cast<char*>(out.data()), 32, "%lld_%lld_%lld_%lld",
                       dims[0], dims[1], dims[2], dims[3]);
    out.resize(len);
    return out;
}

string testNameGenerator(const ::testing::TestParamInfo<clamp_params> info) {
    stringstream ss;
    ss << "size_" << pd4(info.param.size_) << "_in_" << info.param.in_type_
       << "_lo_" << info.param.lo_type_ << "_hi_" << info.param.hi_type_;
    return ss.str();
}

typedef Clamp<double> ClampFloatingPoint;

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    SmallDims, ClampFloatingPoint,
    ::testing::Values(
                      clamp_params(dim4(10), f32, f32, f32, f32),
                      clamp_params(dim4(10), f64, f32, f32, f64),
                      clamp_params(dim4(10), f16, f32, f32, f32),
                      clamp_params(dim4(10), f64, f64, f64, f64),
                      clamp_params(dim4(10), f16, f16, f16, f16),
                      clamp_params(dim4(10), s32, f32, f32, f32),
                      clamp_params(dim4(10), u32, f32, f32, f32),
                      clamp_params(dim4(10), u8,  f32, f32, f32),
                      clamp_params(dim4(10), b8,  f32, f32, f32),
                      clamp_params(dim4(10), s64, f32, f32, f32),
                      clamp_params(dim4(10), u64, f32, f32, f32),
                      clamp_params(dim4(10), s16, f32, f32, f32),
                      clamp_params(dim4(10), u16, f32, f32, f32),

                      clamp_params(dim4(10, 10), f32, f32, f32, f32),
                      clamp_params(dim4(10, 10), f64, f32, f32, f64),
                      clamp_params(dim4(10, 10), f16, f32, f32, f32),
                      clamp_params(dim4(10, 10), f64, f64, f64, f64),
                      clamp_params(dim4(10, 10), f16, f16, f16, f16),

                      clamp_params(dim4(10, 10, 10), f32, f32, f32, f32),
                      clamp_params(dim4(10, 10, 10), f64, f32, f32, f64),
                      clamp_params(dim4(10, 10, 10), f16, f32, f32, f32),
                      clamp_params(dim4(10, 10, 10), f64, f64, f64, f64),
                      clamp_params(dim4(10, 10, 10), f16, f16, f16, f16)
                      ),
    testNameGenerator);
// clang-format on

TEST_P(ClampFloatingPoint, Basic) {
    clamp_params params = GetParam();
    array out           = clamp(in_, lo_, hi_);
    ASSERT_ARRAYS_NEAR(gold_, out, 1e-5);
}

TEST(Clamp, FloatArrayArray) {
    array in = randu(num, f32);
    array lo = randu(num, f32) / 10;        // Ensure lo <= 0.1
    array hi = 1.0 - randu(num, f32) / 10;  // Ensure hi >= 0.9
    eval(lo, hi);

    vector<float> hout(num), hin(num), hlo(num), hhi(num);
    array out = clamp(in, lo, hi);
    out.host(&hout[0]);
    in.host(&hin[0]);
    lo.host(&hlo[0]);
    hi.host(&hhi[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hhi[i]);
        ASSERT_GE(hout[i], hlo[i]);
        ASSERT_EQ(true,
                  hout[i] == hin[i] || hout[i] == hlo[i] || hout[i] == hhi[i]);
    }
}

TEST(Clamp, FloatArrayScalar) {
    array in = randu(num, f32);
    array lo = randu(num, f32) / 10;  // Ensure lo <= 0.1
    float hi = 0.9;

    vector<float> hout(num), hin(num), hlo(num);
    array out = clamp(in, lo, hi);

    out.host(&hout[0]);
    in.host(&hin[0]);
    lo.host(&hlo[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hi);
        ASSERT_GE(hout[i], hlo[i]);
        ASSERT_EQ(true,
                  hout[i] == hin[i] || hout[i] == hlo[i] || hout[i] == hi);
    }
}

TEST(Clamp, FloatScalarArray) {
    array in = randu(num, f32);
    float lo = 0.1;
    array hi = 1.0 - randu(num, f32) / 10;  // Ensure hi >= 0.9

    vector<float> hout(num), hin(num), hhi(num);
    array out = clamp(in, lo, hi);

    out.host(&hout[0]);
    in.host(&hin[0]);
    hi.host(&hhi[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hhi[i]);
        ASSERT_GE(hout[i], lo);
        ASSERT_EQ(true,
                  hout[i] == hin[i] || hout[i] == lo || hout[i] == hhi[i]);
    }
}

TEST(Clamp, FloatScalarScalar) {
    array in = randu(num, f32);
    float lo = 0.1;
    float hi = 0.9;

    vector<float> hout(num), hin(num);
    array out = clamp(in, lo, hi);

    out.host(&hout[0]);
    in.host(&hin[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hi);
        ASSERT_GE(hout[i], lo);
        ASSERT_EQ(true, hout[i] == hin[i] || hout[i] == lo || hout[i] == hi);
    }
}
