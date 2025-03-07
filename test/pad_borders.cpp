/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using std::vector;

template<typename T>
class PadBorders : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, char, unsigned char,
                         int, uint, intl, uintl, short,
                         ushort /*, half_float::half*/>
    TestTypes;

TYPED_TEST_SUITE(PadBorders, TestTypes);

template<typename T>
void testPad(const vector<T>& input, const dim4& inDims, const dim4& lbPadding,
             const dim4& ubPadding, const af::borderType btype,
             const vector<T>& gold, const dim4& outDims) {
    SUPPORTED_TYPE_CHECK(T);
    array in(inDims, input.data());
    array out = af::pad(in, lbPadding, ubPadding, btype);
    ASSERT_VEC_ARRAY_EQ(gold, outDims, out);
}

TYPED_TEST(PadBorders, Zero) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0), AF_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, ClampToEdge) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2,
                2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0),
            AF_PAD_CLAMP_TO_EDGE,
            vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, SymmetricOverEdge) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 0, 2, 3, 2, 2, 0, 3, 5, 2,
                2, 0, 4, 7, 3, 3, 0, 5, 9, 1, 1, 0,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0), AF_PAD_SYM,
            vector<TypeParam>({
                3, 2, 2, 3, 2, 2, 0, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 3, 2, 2, 3, 2, 2, 0, 0, 2, 5, 3, 3, 5, 2, 2,
                0, 0, 2, 7, 4, 4, 7, 3, 3, 0, 0, 3, 9, 5, 5, 9, 1, 1, 0, 0, 1,
                9, 5, 5, 9, 1, 1, 0, 0, 1, 7, 4, 4, 7, 3, 3, 0, 0, 3,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, Periodic) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 0, 2, 3, 2, 2, 0, 3, 5, 2,
                2, 0, 4, 7, 3, 3, 0, 5, 9, 1, 1, 0,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0), AF_PAD_PERIODIC,
            vector<TypeParam>({
                3, 0, 4, 7, 3, 3, 0, 4, 7, 1, 0, 5, 9, 1, 1, 0, 5, 9, 1, 0, 1,
                1, 1, 1, 0, 1, 1, 2, 0, 2, 3, 2, 2, 0, 2, 3, 2, 0, 3, 5, 2, 2,
                0, 3, 5, 3, 0, 4, 7, 3, 3, 0, 4, 7, 1, 0, 5, 9, 1, 1, 0, 5, 9,
                1, 0, 1, 1, 1, 1, 0, 1, 1, 2, 0, 2, 3, 2, 2, 0, 2, 3,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, BeginOnly) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(0, 2, 0, 0), AF_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(7, 9));
}

TYPED_TEST(PadBorders, EndOnly) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(0, 2, 0, 0), dim4(2, 2, 0, 0), AF_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(7, 9));
}

TYPED_TEST(PadBorders, BeginCorner) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(0, 0, 0, 0), AF_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            }),
            dim4(7, 7));
}

TYPED_TEST(PadBorders, EndCorner) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(0, 0, 0, 0), dim4(2, 2, 0, 0), AF_PAD_ZERO,
            vector<TypeParam>({
                1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(7, 7));
}

TEST(PadBorders, NegativePadding) {
    af_array dummyIn  = 0;
    af_array dummyOut = 0;
    dim_t ldims[4]    = {-1, 1, 0, 1};
    dim_t udims[4]    = {-1, 1, 0, 1};
    ASSERT_EQ(AF_ERR_SIZE,
              af_pad(&dummyOut, dummyIn, 4, ldims, 4, udims, AF_PAD_ZERO));
}

TEST(PadBorders, NegativeNDims) {
    af_array dummyIn  = 0;
    af_array dummyOut = 0;
    dim_t ldims[4]    = {1, 1, 0, 1};
    dim_t udims[4]    = {1, 1, 0, 1};
    ASSERT_EQ(AF_ERR_SIZE,
              af_pad(&dummyOut, dummyIn, -1, ldims, 4, udims, AF_PAD_ZERO));
}

TEST(PadBorders, InvalidPadType) {
    af_array dummyIn  = 0;
    af_array dummyOut = 0;
    dim_t ldims[4]    = {1, 1, 0, 1};
    dim_t udims[4]    = {1, 1, 0, 1};
    ASSERT_EQ(AF_ERR_ARG, af_pad(&dummyOut, dummyIn, 4, ldims, 4, udims,
                                 (af_border_type)4));
}
