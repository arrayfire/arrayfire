/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/index.h>
#include <af/traits.hpp>

#include <complex>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::join;
using af::randu;
using af::seq;
using af::sum;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Join : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int,
                         intl, uintl, char, unsigned char, short, ushort,
                         half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Join, TestTypes);

template<typename T>
void joinTest(string pTestFile, const unsigned dim, const unsigned in0,
              const unsigned in1, const unsigned resultIdx,
              bool isSubRef = false, const vector<af_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 i0dims = numDims[in0];
    dim4 i1dims = numDims[in1];

    af_array in0Array  = 0;
    af_array in1Array  = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(in[in0].front()),
                                       i0dims.ndims(), i0dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&in0Array, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&in0Array, &(in[in0].front()),
                                       i0dims.ndims(), i0dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(in[in1].front()),
                                       i1dims.ndims(), i1dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&in1Array, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&in1Array, &(in[in1].front()),
                                       i1dims.ndims(), i1dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_join(&outArray, dim, in0Array, in1Array));

    dim4 goldDims = i0dims;
    goldDims[dim] = i0dims[dim] + i1dims[dim];

    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], goldDims, outArray);

    if (in0Array != 0) af_release_array(in0Array);
    if (in1Array != 0) af_release_array(in1Array);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

#define JOIN_INIT(desc, file, dim, in0, in1, resultIdx)                        \
    TYPED_TEST(Join, desc) {                                                   \
        joinTest<TypeParam>(string(TEST_DIR "/join/" #file ".test"), dim, in0, \
                            in1, resultIdx);                                   \
    }

JOIN_INIT(JoinBig0, join_big, 0, 0, 1, 0);
JOIN_INIT(JoinBig1, join_big, 1, 0, 2, 1);
JOIN_INIT(JoinBig2, join_big, 2, 0, 3, 2);

JOIN_INIT(JoinSmall0, join_small, 0, 0, 1, 0);
JOIN_INIT(JoinSmall1, join_small, 1, 0, 2, 1);
JOIN_INIT(JoinSmall2, join_small, 2, 0, 3, 2);

TEST(Join, JoinLargeDim) {
    using af::constant;
    using af::deviceGC;
    using af::span;

    // const int nx = 32;
    const int nx = 1;
    const int ny = 4 * 1024 * 1024;
    const int nw = 4 * 1024 * 1024;

    deviceGC();
    {
        array in         = randu(nx, ny, u8);
        array joined     = join(0, in, in);
        dim4 in_dims     = in.dims();
        dim4 joined_dims = joined.dims();

        ASSERT_EQ(2 * in_dims[0], joined_dims[0]);
        ASSERT_EQ(0.f, sum<float>((joined(0, span) - joined(1, span)).as(f32)));

        array in2 = constant(1, (dim_t)nx, (dim_t)ny, (dim_t)2, (dim_t)nw, u8);
        joined    = join(3, in, in);
        in_dims   = in.dims();
        joined_dims = joined.dims();
        ASSERT_EQ(2 * in_dims[3], joined_dims[3]);
    }
}

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Join, CPP) {
    const unsigned resultIdx = 2;
    const unsigned dim       = 2;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/join/join_big.test"),
                                 numDims, in, tests);

    dim4 i0dims = numDims[0];
    dim4 i1dims = numDims[3];

    array input0(i0dims, &(in[0].front()));
    array input1(i1dims, &(in[3].front()));

    array output = join(dim, input0, input1);

    dim4 goldDims = i0dims;
    goldDims[dim] = i0dims[dim] + i1dims[dim];

    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], goldDims, output);
}

TEST(JoinMany0, CPP) {
    array a0 = randu(10, 5);
    array a1 = randu(20, 5);
    array a2 = randu(5, 5);

    array output = join(0, a0, a1, a2);
    array gold   = join(0, a0, join(0, a1, a2));

    ASSERT_EQ(sum<float>(output - gold), 0);
}

TEST(JoinMany1, CPP) {
    array a0 = randu(20, 200);
    array a1 = randu(20, 400);
    array a2 = randu(20, 10);
    array a3 = randu(20, 100);

    int dim      = 1;
    array output = join(dim, a0, a1, a2, a3);
    array gold   = join(dim, a0, join(dim, a1, join(dim, a2, a3)));
    ASSERT_EQ(sum<float>(output - gold), 0);
}

TEST(Join, DifferentSizes) {
    array a = seq(10);
    array b = seq(11);
    array c = seq(12);

    array d = join(0, a, b, c);

    vector<float> ha(10);
    vector<float> hb(11);
    vector<float> hc(12);

    for (size_t i = 0; i < ha.size(); i++) { ha[i] = i; }
    for (size_t i = 0; i < hb.size(); i++) { hb[i] = i; }
    for (size_t i = 0; i < hc.size(); i++) { hc[i] = i; }
    vector<float> hgold(10 + 11 + 12);
    vector<float>::iterator it = copy(ha.begin(), ha.end(), hgold.begin());
    it                         = copy(hb.begin(), hb.end(), it);
    it                         = copy(hc.begin(), hc.end(), it);

    ASSERT_VEC_ARRAY_EQ(hgold, dim4(10 + 11 + 12), d);
}

TEST(Join, SameSize) {
    array a = seq(10);
    array b = seq(10);
    array c = seq(10);

    array d = join(0, a, b, c);

    vector<float> ha(10);
    vector<float> hb(10);
    vector<float> hc(10);

    for (size_t i = 0; i < ha.size(); i++) { ha[i] = i; }
    for (size_t i = 0; i < hb.size(); i++) { hb[i] = i; }
    for (size_t i = 0; i < hc.size(); i++) { hc[i] = i; }
    vector<float> hgold(10 + 10 + 10);
    vector<float>::iterator it = copy(ha.begin(), ha.end(), hgold.begin());
    it                         = copy(hb.begin(), hb.end(), it);
    it                         = copy(hc.begin(), hc.end(), it);

    ASSERT_VEC_ARRAY_EQ(hgold, dim4(10 + 10 + 10), d);
}

TEST(Join, ManyEmpty) {
    array gold = af::constant(0, 15, 5);
    array a    = af::randn(5, 5);
    array e;
    array c  = af::randn(10, 5);
    array ee = af::join(0, e, e);
    ASSERT_EQ(ee.elements(), 0);
    array eee = af::join(0, e, e, e);
    ASSERT_EQ(eee.elements(), 0);

    array eeac                     = af::join(0, e, e, a, c);
    array eace                     = af::join(0, e, a, c, e);
    array acee                     = af::join(0, a, c, e, e);
    gold(af::seq(0, 4), af::span)  = a;
    gold(af::seq(5, 14), af::span) = c;
    ASSERT_ARRAYS_EQ(gold, eeac);
    ASSERT_ARRAYS_EQ(gold, eace);
    ASSERT_ARRAYS_EQ(gold, acee);
}
