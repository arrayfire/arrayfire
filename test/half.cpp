/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#define GTEST_LINKED_AS_SHARED_LIBRARY 1
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <../extern/half/include/half.hpp>
#include <testHelpers.hpp>

using af::array;
using af::constant;
using af::half;
using std::vector;

TEST(Half, print) {
    SUPPORTED_TYPE_CHECK(af_half);
    array aa = af::constant(3.14, 3, 3, f16);
    array bb = af::constant(2, 3, 3, f16);
    af_print(aa);
}

struct convert_params {
    af_dtype from, to;
    double value;
    convert_params(af_dtype f, af_dtype t, double v)
        : from(f), to(t), value(v) {}
};

class HalfConvert : public ::testing::TestWithParam<convert_params> {};

INSTANTIATE_TEST_CASE_P(ToF16, HalfConvert,
                        ::testing::Values(convert_params(f32, f16, 10),
                                          convert_params(f64, f16, 10),
                                          convert_params(s32, f16, 10),
                                          convert_params(u32, f16, 10),
                                          convert_params(u8, f16, 10),
                                          convert_params(s64, f16, 10),
                                          convert_params(u64, f16, 10),
                                          convert_params(s16, f16, 10),
                                          convert_params(u16, f16, 10),
                                          convert_params(f16, f16, 10)));

INSTANTIATE_TEST_CASE_P(FromF16, HalfConvert,
                        ::testing::Values(convert_params(f16, f32, 10),
                                          convert_params(f16, f64, 10),
                                          convert_params(f16, s32, 10),
                                          convert_params(f16, u32, 10),
                                          // causes compilation failures with
                                          // nvrtc
                                          // convert_params(f16, u8, 10),
                                          convert_params(f16, s64, 10),
                                          convert_params(f16, u64, 10),
                                          convert_params(f16, s16, 10),
                                          convert_params(f16, u16, 10),
                                          convert_params(f16, f16, 10)));

TEST_P(HalfConvert, convert) {
    SUPPORTED_TYPE_CHECK(af_half);
    convert_params params = GetParam();

    array from = af::constant(params.value, 3, 3, params.from);
    array to   = from.as(params.to);

    ASSERT_EQ(from.type(), params.from);
    ASSERT_EQ(to.type(), params.to);

    array gold = af::constant(params.value, 3, 3, params.to);
    ASSERT_ARRAYS_EQ(gold, to);
}

TEST(Half, arith) {
    SUPPORTED_TYPE_CHECK(af_half);
    array aa = af::constant(3.14, 3, 3, f16);
    array bb = af::constant(1, 3, 3, f16);

    array gold   = constant(4.14, 3, 3, f16);
    array result = bb + aa;

    ASSERT_ARRAYS_EQ(gold, result);
}
