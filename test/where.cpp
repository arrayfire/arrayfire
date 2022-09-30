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
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using af::allTrue;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::randu;
using af::range;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Where : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, int, uint, intl, uintl,
                         char, uchar, short, ushort>
    TestTypes;
TYPED_TEST_SUITE(Where, TestTypes);

template<typename T>
void whereTest(string pTestFile, bool isSubRef = false,
               const vector<af_seq> seqv = vector<af_seq>()) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);
    dim4 dims = numDims[0];

    vector<T> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(), convert_to<T, int>);

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    // Compare result
    vector<uint> currGoldBar(tests[0].begin(), tests[0].end());

    // Run sum
    ASSERT_SUCCESS(af_where(&outArray, inArray));

    ASSERT_VEC_ARRAY_EQ(currGoldBar, dim4(tests[0].size()), outArray);

    if (inArray != 0) af_release_array(inArray);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

#define WHERE_TESTS(T)                                      \
    TEST(Where, Test_##T) {                                 \
        whereTest<T>(string(TEST_DIR "/where/where.test")); \
    }

TYPED_TEST(Where, BasicC) {
    whereTest<TypeParam>(string(TEST_DIR "/where/where.test"));
}

//////////////////////////////////// CPP /////////////////////////////////
//
TYPED_TEST(Where, CPP) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(string(TEST_DIR "/where/where.test"), numDims,
                             data, tests);
    dim4 dims = numDims[0];

    vector<float> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(),
              convert_to<float, int>);

    array input(dims, &in.front(), afHost);
    array output = where(input);

    // Compare result
    vector<uint> currGoldBar(tests[0].begin(), tests[0].end());

    ASSERT_VEC_ARRAY_EQ(currGoldBar, dim4(tests[0].size()), output);
}

TEST(Where, MaxDim) {
    const size_t largeDim = 65535 * 32 + 2;

    array input  = range(dim4(1, largeDim), 1);
    array output = where(input % 2 == 0);
    array gold   = 2 * range(largeDim / 2);
    ASSERT_ARRAYS_EQ(gold.as(u32), output);

    input  = range(dim4(1, 1, 1, largeDim), 3);
    output = where(input % 2 == 0);
    ASSERT_ARRAYS_EQ(gold.as(u32), output);
}

TEST(Where, ISSUE_1259) {
    array a       = randu(10, 10, 10);
    array indices = where(a > 2);
    ASSERT_EQ(indices.elements(), 0);
}
