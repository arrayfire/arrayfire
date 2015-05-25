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
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

template<typename T>
class Reorder : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back(af_make_seq(0, 4, 1));
            subMat0.push_back(af_make_seq(2, 6, 1));
            subMat0.push_back(af_make_seq(0, 2, 1));
        }
        vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Reorder, TestTypes);

template<typename T>
void reorderTest(string pTestFile, const unsigned resultIdx,
                 const uint x, const uint y, const uint z, const uint w,
                 bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, int>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];

    af_array inArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_reorder(&outArray, inArray, x, y, z, w));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define REORDER_INIT(desc, file, resultIdx, x, y, z, w)                                        \
    TYPED_TEST(Reorder, desc)                                                                  \
    {                                                                                       \
        reorderTest<TypeParam>(string(TEST_DIR"/reorder/"#file".test"), resultIdx, x, y, z, w);   \
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
    REORDER_INIT(Reorder1302, reorder4d,10, 1, 3, 0, 2);
    REORDER_INIT(Reorder1320, reorder4d,11, 1, 3, 2, 0);

    REORDER_INIT(Reorder2103, reorder4d,12, 2, 1, 0, 3);
    REORDER_INIT(Reorder2130, reorder4d,13, 2, 1, 3, 0);
    REORDER_INIT(Reorder2013, reorder4d,14, 2, 0, 1, 3);
    REORDER_INIT(Reorder2031, reorder4d,15, 2, 0, 3, 1);
    REORDER_INIT(Reorder2310, reorder4d,16, 2, 3, 1, 0);
    REORDER_INIT(Reorder2301, reorder4d,17, 2, 3, 0, 1);

    REORDER_INIT(Reorder3120, reorder4d,18, 3, 1, 2, 0);
    REORDER_INIT(Reorder3102, reorder4d,19, 3, 1, 0, 2);
    REORDER_INIT(Reorder3210, reorder4d,20, 3, 2, 1, 0);
    REORDER_INIT(Reorder3201, reorder4d,21, 3, 2, 0, 1);
    REORDER_INIT(Reorder3012, reorder4d,22, 3, 0, 1, 2);
    REORDER_INIT(Reorder3021, reorder4d,23, 3, 0, 2, 1);

////////////////////////////////// CPP ///////////////////////////////////
//
TEST(Reorder, CPP)
{
    if (noDoubleTests<float>()) return;

    const unsigned resultIdx = 0;
    const unsigned x = 0;
    const unsigned y = 1;
    const unsigned z = 2;
    const unsigned w = 3;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, int>(string(TEST_DIR"/reorder/reorder4d.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];

    af::array input(idims, &(in[0].front()));
    af::array output = af::reorder(input, x, y, z, w);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;
}

