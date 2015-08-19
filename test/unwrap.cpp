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
class Unwrap : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int, intl, uintl, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Unwrap, TestTypes);

template<typename T>
void unwrapTest(string pTestFile, const unsigned resultIdx,
        const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px, const dim_t py)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, int>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];

    af_array inArray = 0;
    af_array outArray = 0;
    af_array outArrayT = 0;
    af_array outArray2 = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_unwrap(&outArray , inArray, wx, wy, sx, sy, px, py, true ));
    ASSERT_EQ(AF_SUCCESS, af_unwrap(&outArrayT, inArray, wx, wy, sx, sy, px, py, false));
    ASSERT_EQ(AF_SUCCESS, af_transpose(&outArray2, outArrayT, false));

    size_t nElems = tests[resultIdx].size();
    std::vector<T> outData(nElems);

    // Compare is_column == true results
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)&outData[0], outArray));
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    // Compare is_column == false results
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)&outData[0], outArray2));
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    if(inArray   != 0) af_release_array(inArray);
    if(outArray  != 0) af_release_array(outArray);
    if(outArrayT != 0) af_release_array(outArrayT);
    if(outArray2 != 0) af_release_array(outArray2);
}

#define UNWRAP_INIT(desc, file, resultIdx, wx, wy, sx, sy, px,py)       \
    TYPED_TEST(Unwrap, desc)                                            \
    {                                                                   \
        unwrapTest<TypeParam>(string(TEST_DIR"/unwrap/"#file".test"),   \
                              resultIdx, wx, wy, sx, sy, px, py);       \
    }

    UNWRAP_INIT(UnwrapSmall00, unwrap_small,  0,  3,  3,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall01, unwrap_small,  1,  3,  3,  1,  1,  1,  1);
    UNWRAP_INIT(UnwrapSmall02, unwrap_small,  2,  3,  3,  1,  1,  2,  2);
    UNWRAP_INIT(UnwrapSmall03, unwrap_small,  3,  3,  3,  2,  2,  0,  0);
    UNWRAP_INIT(UnwrapSmall04, unwrap_small,  4,  3,  3,  2,  2,  1,  1);
    UNWRAP_INIT(UnwrapSmall05, unwrap_small,  5,  3,  3,  2,  2,  2,  2);
    UNWRAP_INIT(UnwrapSmall06, unwrap_small,  6,  3,  3,  3,  3,  0,  0);
    UNWRAP_INIT(UnwrapSmall07, unwrap_small,  7,  3,  3,  3,  3,  1,  1);
    UNWRAP_INIT(UnwrapSmall08, unwrap_small,  8,  3,  3,  3,  3,  2,  2);
    UNWRAP_INIT(UnwrapSmall09, unwrap_small,  9,  4,  4,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall10, unwrap_small, 10,  4,  4,  1,  1,  1,  1);
    UNWRAP_INIT(UnwrapSmall11, unwrap_small, 11,  4,  4,  1,  1,  2,  2);
    UNWRAP_INIT(UnwrapSmall12, unwrap_small, 12,  4,  4,  1,  1,  3,  3);
    UNWRAP_INIT(UnwrapSmall13, unwrap_small, 13,  4,  4,  2,  2,  0,  0);
    UNWRAP_INIT(UnwrapSmall14, unwrap_small, 14,  4,  4,  2,  2,  1,  1);
    UNWRAP_INIT(UnwrapSmall15, unwrap_small, 15,  4,  4,  2,  2,  2,  2);
    UNWRAP_INIT(UnwrapSmall16, unwrap_small, 16,  4,  4,  2,  2,  3,  3);
    UNWRAP_INIT(UnwrapSmall17, unwrap_small, 17,  4,  4,  4,  4,  0,  0);
    UNWRAP_INIT(UnwrapSmall18, unwrap_small, 18,  4,  4,  4,  4,  1,  1);
    UNWRAP_INIT(UnwrapSmall19, unwrap_small, 19,  4,  4,  4,  4,  2,  2);
    UNWRAP_INIT(UnwrapSmall20, unwrap_small, 20,  4,  4,  4,  4,  3,  3);
    UNWRAP_INIT(UnwrapSmall21, unwrap_small, 21,  5,  5,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall22, unwrap_small, 22,  5,  5,  1,  1,  1,  1);
    UNWRAP_INIT(UnwrapSmall23, unwrap_small, 23,  5,  5,  5,  5,  0,  0);
    UNWRAP_INIT(UnwrapSmall24, unwrap_small, 24,  5,  5,  5,  5,  1,  1);
    UNWRAP_INIT(UnwrapSmall25, unwrap_small, 25,  8,  8,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall26, unwrap_small, 26,  8,  8,  1,  1,  7,  7);
    UNWRAP_INIT(UnwrapSmall27, unwrap_small, 27,  8,  8,  8,  8,  0,  0);
    UNWRAP_INIT(UnwrapSmall28, unwrap_small, 28,  8,  8,  8,  8,  7,  7);
    UNWRAP_INIT(UnwrapSmall29, unwrap_small, 29, 12, 12,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall30, unwrap_small, 30, 12, 12,  1,  1,  2,  2);
    UNWRAP_INIT(UnwrapSmall31, unwrap_small, 31, 12, 12, 12, 12,  0,  0);
    UNWRAP_INIT(UnwrapSmall32, unwrap_small, 32, 12, 12, 12, 12,  2,  2);
    UNWRAP_INIT(UnwrapSmall33, unwrap_small, 33, 16, 16,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall34, unwrap_small, 34, 16, 16, 16, 16,  0,  0);
    UNWRAP_INIT(UnwrapSmall35, unwrap_small, 35, 16, 16, 16, 16, 15, 15);
    UNWRAP_INIT(UnwrapSmall36, unwrap_small, 36, 31, 31,  8,  8, 15, 15);
    UNWRAP_INIT(UnwrapSmall37, unwrap_small, 37,  8, 12,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall38, unwrap_small, 38,  8, 12,  1,  1,  7, 11);
    UNWRAP_INIT(UnwrapSmall39, unwrap_small, 39,  8, 12,  8, 12,  0,  0);
    UNWRAP_INIT(UnwrapSmall40, unwrap_small, 40,  8, 12,  8, 12,  7, 11);
    UNWRAP_INIT(UnwrapSmall41, unwrap_small, 41, 15, 10,  1,  1,  0,  0);
    UNWRAP_INIT(UnwrapSmall42, unwrap_small, 42, 15, 10,  1,  1, 14,  9);
    UNWRAP_INIT(UnwrapSmall43, unwrap_small, 43, 15, 10, 15, 10,  0,  0);

   // FIXME: This test is faulty after fixing the copy paste errors in unwrap
   // UNWRAP_INIT(UnwrapSmall44, unwrap_small, 44, 15, 10, 15, 10, 14,  9);

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Unwrap, CPP)
{
    if (noDoubleTests<float>()) return;

    const unsigned resultIdx = 20;
    const unsigned wx = 4;
    const unsigned wy = 4;
    const unsigned sx = 4;
    const unsigned sy = 4;
    const unsigned px = 3;
    const unsigned py = 3;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, int>(string(TEST_DIR"/unwrap/unwrap_small.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array output = af::unwrap(input, wx, wy, sx, sy, px, py);

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
