/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <af/array.h>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::endl;
using af::allTrue;
using af::array;
using af::cfloat;
using af::cdouble;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::randu;
using af::range;

template<typename T>
class Where : public ::testing::Test { };

typedef ::testing::Types< float, double, cfloat, cdouble, int, uint, intl, uintl, char, uchar, short, ushort> TestTypes;
TYPED_TEST_CASE(Where, TestTypes);

template<typename T>
void whereTest(string pTestFile, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    if (noDoubleTests<T>()) return;

    vector<dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    dim4 dims       = numDims[0];

    vector<T> in(data[0].begin(), data[0].end());

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) dtype_traits<T>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
    } else {

        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) dtype_traits<T>::af_type));
    }

    // Compare result
    vector<uint> currGoldBar(tests[0].begin(), tests[0].end());

    // Run sum
    ASSERT_EQ(AF_SUCCESS, af_where(&outArray, inArray));

    // Get result
    size_t nElems = currGoldBar.size();
    vector<uint> outData(nElems);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr(&outData.front(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                        << endl;
    }

    if(inArray   != 0) af_release_array(inArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define WHERE_TESTS(T)                          \
    TEST(Where,Test_##T)                        \
    {                                           \
        whereTest<T>(                           \
            string(TEST_DIR"/where/where.test") \
            );                                  \
    }                                           \

TYPED_TEST(Where, BasicC)
{
    whereTest<TypeParam>(string(TEST_DIR"/where/where.test") );
}

//////////////////////////////////// CPP /////////////////////////////////
//
TYPED_TEST(Where, CPP)
{
    if (noDoubleTests<TypeParam>()) return;

    vector<dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (string(TEST_DIR"/where/where.test"),numDims,data,tests);
    dim4 dims       = numDims[0];

    vector<float> in(data[0].begin(), data[0].end());
    array input(dims, &in.front(), afHost);
    array output = where(input);

    // Compare result
    vector<uint> currGoldBar(tests[0].begin(), tests[0].end());

    // Get result
    size_t nElems = currGoldBar.size();
    vector<uint> outData(nElems);
    output.host((void*)&(outData.front()));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                        << endl;
    }
}

TEST(Where, MaxDim)
{
    const size_t largeDim = 65535 * 32 + 2;

    array input = range(dim4(1, largeDim), 1);
    array output = where(input % 2 == 0);
    array gold = 2 * range(largeDim/2);
    ASSERT_TRUE(allTrue<bool>(output == gold));

    input = range(dim4(1, 1, 1, largeDim), 3);
    output = where(input % 2 == 0);
    ASSERT_TRUE(allTrue<bool>(output == gold));
}

TEST(Where, ISSUE_1259)
{
    array a = randu(10, 10, 10);
    array indices = where(a > 2);
    ASSERT_EQ(indices.elements(), 0);
}
