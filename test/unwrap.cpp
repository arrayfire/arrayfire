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
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Unwrap, TestTypes);

template<typename T>
void unwrapTest(string pTestFile, const unsigned resultIdx, const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, int>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];

    af_array inArray = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_unwrap(&outArray, inArray, wx, wy, sx, sy));

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
}

#define UNWRAP_INIT(desc, file, resultIdx, x, y, z, w)                                          \
    TYPED_TEST(Unwrap, desc)                                                                    \
    {                                                                                           \
        unwrapTest<TypeParam>(string(TEST_DIR"/unwrap/"#file".test"), resultIdx, x, y, z, w);   \
    }

    //UNWRAP_INIT(Unwrap00, unwrap,  0,  3, 3, 1, 1);
    //UNWRAP_INIT(Unwrap01, unwrap,  1,  4, 4, 1, 1);
    //UNWRAP_INIT(Unwrap02, unwrap,  2,  5, 5, 1, 1);
    //UNWRAP_INIT(Unwrap03, unwrap,  3,  5, 5, 5, 5);
    //UNWRAP_INIT(Unwrap04, unwrap,  4,  6, 6, 1, 1);
    //UNWRAP_INIT(Unwrap05, unwrap,  5,  9, 9, 1, 1);
    //UNWRAP_INIT(Unwrap06, unwrap,  6, 16,16, 1, 1);
    //UNWRAP_INIT(Unwrap07, unwrap,  7, 32,32, 1, 1);
    //UNWRAP_INIT(Unwrap08, unwrap,  8,  8, 2, 1, 1);
    //UNWRAP_INIT(Unwrap09, unwrap,  9,  4, 5, 1, 1);
    //UNWRAP_INIT(Unwrap10, unwrap, 10,  8, 4, 1, 1);
    //UNWRAP_INIT(Unwrap11, unwrap, 11, 10, 5, 1, 1);
    //UNWRAP_INIT(Unwrap12, unwrap, 12, 32, 2, 1, 1);
    //UNWRAP_INIT(Unwrap13, unwrap, 13,  2,50, 1, 1);
    //UNWRAP_INIT(Unwrap14, unwrap, 14, 90, 4, 1, 1);

    UNWRAP_INIT(UnwrapSmall00, unwrap_small,  0,  3, 3, 1, 1);
    UNWRAP_INIT(UnwrapSmall01, unwrap_small,  1,  4, 4, 1, 1);
    UNWRAP_INIT(UnwrapSmall02, unwrap_small,  2,  5, 5, 1, 1);
    UNWRAP_INIT(UnwrapSmall03, unwrap_small,  3,  6, 6, 1, 1);
    UNWRAP_INIT(UnwrapSmall04, unwrap_small,  4,  8, 8, 1, 1);
    UNWRAP_INIT(UnwrapSmall05, unwrap_small,  5, 12,12, 1, 1);
    UNWRAP_INIT(UnwrapSmall06, unwrap_small,  6,  5, 3, 1, 1);
    UNWRAP_INIT(UnwrapSmall07, unwrap_small,  7, 10, 4, 1, 1);
    UNWRAP_INIT(UnwrapSmall08, unwrap_small,  8, 15,15, 1, 1);
    UNWRAP_INIT(UnwrapSmall09, unwrap_small,  9, 12,10, 1, 1);

    UNWRAP_INIT(UnwrapSmall10, unwrap_small, 10,  3, 3,  3, 3);
    UNWRAP_INIT(UnwrapSmall11, unwrap_small, 11,  4, 4,  4, 4);
    UNWRAP_INIT(UnwrapSmall12, unwrap_small, 12,  5, 5,  5, 5);
    UNWRAP_INIT(UnwrapSmall13, unwrap_small, 13,  6, 6,  6, 6);
    UNWRAP_INIT(UnwrapSmall14, unwrap_small, 14,  8, 8,  8, 8);
    UNWRAP_INIT(UnwrapSmall15, unwrap_small, 15, 12,12, 12,12);
    UNWRAP_INIT(UnwrapSmall16, unwrap_small, 16,  5, 3,  5, 3);
    UNWRAP_INIT(UnwrapSmall17, unwrap_small, 17, 10, 4, 10, 4);
    UNWRAP_INIT(UnwrapSmall18, unwrap_small, 18, 15,15, 15,15);
    UNWRAP_INIT(UnwrapSmall19, unwrap_small, 19, 12,10, 12,10);

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Unwrap, CPP)
{
    if (noDoubleTests<float>()) return;

    const unsigned resultIdx = 0;
    const unsigned wx = 3;
    const unsigned wy = 3;
    const unsigned sx = 1;
    const unsigned sy = 1;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, int>(string(TEST_DIR"/unwrap/unwrap_small.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array output = af::unwrap(input, wx, wy, sx, sy);

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

