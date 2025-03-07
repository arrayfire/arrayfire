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
#include <af/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::allTrue;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::reorder;
using af::seq;
using af::span;
using af::tile;
using std::string;
using std::vector;

template<typename T>
class Reorder : public ::testing::Test {
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
                         char, unsigned char, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Reorder, TestTypes);

template<typename T>
void reorderTest(string pTestFile, const unsigned resultIdx, const uint x,
                 const uint y, const uint z, const uint w,
                 bool isSubRef = false, const vector<af_seq> *seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_reorder(&outArray, inArray, x, y, z, w));

    dim4 goldDims(idims[x], idims[y], idims[z], idims[w]);
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], goldDims, outArray);

    if (inArray != 0) af_release_array(inArray);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

#define REORDER_INIT(desc, file, resultIdx, x, y, z, w)                    \
    TYPED_TEST(Reorder, desc) {                                            \
        reorderTest<TypeParam>(string(TEST_DIR "/reorder/" #file ".test"), \
                               resultIdx, x, y, z, w);                     \
    }

REORDER_INIT(Reorder012, reorder, 0, 0, 1, 2, 3);
REORDER_INIT(Reorder021, reorder, 1, 0, 2, 1, 3);
REORDER_INIT(Reorder102, reorder, 2, 1, 0, 2, 3);
REORDER_INIT(Reorder120, reorder, 3, 1, 2, 0, 3);
REORDER_INIT(Reorder201, reorder, 4, 2, 0, 1, 3);
REORDER_INIT(Reorder210, reorder, 5, 2, 1, 0, 3);

REORDER_INIT(Reorder0123, reorder4d, 0, 0, 1, 2, 3);
REORDER_INIT(Reorder0132, reorder4d, 1, 0, 1, 3, 2);
REORDER_INIT(Reorder0213, reorder4d, 2, 0, 2, 1, 3);
REORDER_INIT(Reorder0231, reorder4d, 3, 0, 2, 3, 1);
REORDER_INIT(Reorder0312, reorder4d, 4, 0, 3, 1, 2);
REORDER_INIT(Reorder0321, reorder4d, 5, 0, 3, 2, 1);

REORDER_INIT(Reorder1023, reorder4d, 6, 1, 0, 2, 3);
REORDER_INIT(Reorder1032, reorder4d, 7, 1, 0, 3, 2);
REORDER_INIT(Reorder1203, reorder4d, 8, 1, 2, 0, 3);
REORDER_INIT(Reorder1230, reorder4d, 9, 1, 2, 3, 0);
REORDER_INIT(Reorder1302, reorder4d, 10, 1, 3, 0, 2);
REORDER_INIT(Reorder1320, reorder4d, 11, 1, 3, 2, 0);

REORDER_INIT(Reorder2103, reorder4d, 12, 2, 1, 0, 3);
REORDER_INIT(Reorder2130, reorder4d, 13, 2, 1, 3, 0);
REORDER_INIT(Reorder2013, reorder4d, 14, 2, 0, 1, 3);
REORDER_INIT(Reorder2031, reorder4d, 15, 2, 0, 3, 1);
REORDER_INIT(Reorder2310, reorder4d, 16, 2, 3, 1, 0);
REORDER_INIT(Reorder2301, reorder4d, 17, 2, 3, 0, 1);

REORDER_INIT(Reorder3120, reorder4d, 18, 3, 1, 2, 0);
REORDER_INIT(Reorder3102, reorder4d, 19, 3, 1, 0, 2);
REORDER_INIT(Reorder3210, reorder4d, 20, 3, 2, 1, 0);
REORDER_INIT(Reorder3201, reorder4d, 21, 3, 2, 0, 1);
REORDER_INIT(Reorder3012, reorder4d, 22, 3, 0, 1, 2);
REORDER_INIT(Reorder3021, reorder4d, 23, 3, 0, 2, 1);

////////////////////////////////// CPP ///////////////////////////////////
//
TEST(Reorder, CPP) {
    const unsigned resultIdx = 0;
    const unsigned x         = 0;
    const unsigned y         = 1;
    const unsigned z         = 2;
    const unsigned w         = 3;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/reorder/reorder4d.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];

    array input(idims, &(in[0].front()));
    array output = reorder(input, x, y, z, w);

    dim4 goldDims(idims[x], idims[y], idims[z], idims[w]);
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], goldDims, output);
}

TEST(Reorder, ISSUE_1777) {
    const int m = 5;
    const int n = 4;
    const int k = 3;
    vector<float> h_input(m * n);

    for (int i = 0; i < m * n; i++) { h_input[i] = (float)(i); }

    array a(m, n, &h_input[0]);
    array a_t = tile(a, 1, 1, 3);
    array a_r = reorder(a_t, 0, 2, 1);

    vector<float> h_output(m * n * k);
    a_r.host((void *)&h_output[0]);
    for (int z = 0; z < n; z++) {
        for (int y = 0; y < k; y++) {
            for (int x = 0; x < m; x++) {
                ASSERT_EQ(h_output[z * k * m + y * m + x], h_input[z * m + x]);
            }
        }
    }
}

TEST(Reorder, MaxDim) {
    const size_t largeDim = 65535 * 32 + 1;

    array input  = range(dim4(2, largeDim, 2), 2);
    array output = reorder(input, 2, 1, 0);

    array gold = range(dim4(2, largeDim, 2));

    ASSERT_ARRAYS_EQ(gold, output);
}

TEST(Reorder, InputArrayUnchanged) {
    float h_input[12] = {0.f, 1.f, 2.f, 3.f, 4.f,  5.f,
                         6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    array input(2, 3, 2, h_input);
    array input_reord = reorder(input, 0, 2, 1);

    array input_gold(2, 3, 2, h_input);
    ASSERT_ARRAYS_EQ(input_gold, input);
}
