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
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::af_cfloat;
using af::af_cdouble;

template<typename T>
class Resize : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back({0, 4, 1});
            subMat0.push_back({2, 6, 1});
            subMat0.push_back({0, 2, 1});
        }
        vector<af_seq> subMat0;
};

template<typename T>
class ResizeI : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back({0, 4, 1});
            subMat0.push_back({2, 6, 1});
            subMat0.push_back({0, 2, 1});

            subMat1.push_back({0, 5, 1});
            subMat1.push_back({0, 5, 1});
            subMat1.push_back({0, 2, 1});
        }
        vector<af_seq> subMat0;
        vector<af_seq> subMat1;
};

// create a list of types to be tested
typedef ::testing::Types<float, double> TestTypesF;
typedef ::testing::Types<int, unsigned, unsigned char> TestTypesI;

// register the type list
TYPED_TEST_CASE(Resize, TestTypesF);
TYPED_TEST_CASE(ResizeI, TestTypesI);

template<typename T>
void resizeTest(string pTestFile, const unsigned resultIdx, const dim_type odim0, const dim_type odim1, const af_interp_type method, bool isSubRef = false, const vector<af_seq> * seqv = nullptr)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T>>   in;
    vector<vector<T>>   tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    af::dim4 dims = numDims[0];

    af_array inArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;
    if (isSubRef) {

        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_resize(&outArray, inArray, odim0, odim1, method));

    // Get result
    af::dim4 odims(odim0, odim1, dims[2], dims[3]);
    T* outData = new T[odims.elements()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(tests[resultIdx][elIter], outData[elIter], 0.0001) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_destroy_array(inArray);
    if(outArray  != 0) af_destroy_array(outArray);
    if(tempArray != 0) af_destroy_array(tempArray);
}

TYPED_TEST(Resize, Resize3CSquareUpNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 0, 16, 16, AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize3CSquareUpLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 1, 16, 16, AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize3CSquareDownNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 2, 4, 4, AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize3CSquareDownLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 3, 4, 4, AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize3CSquareUpNearestSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 4, 10, 10, AF_INTERP_NEAREST,
                          true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize3CSquareUpLinearSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 5, 10, 10, AF_INTERP_BILINEAR,
                          true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize3CSquareDownNearestSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 6, 3, 3, AF_INTERP_NEAREST,
                          true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize3CSquareDownLinearSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 7, 3, 3, AF_INTERP_BILINEAR,
                          true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize1CRectangleUpNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/rectangle.test"), 0, 12, 16, AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CRectangleUpLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/rectangle.test"), 1, 12, 16, AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize1CRectangleDownNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/rectangle.test"), 2, 6, 2, AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CRectangleDownLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/rectangle.test"), 3, 6, 2, AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize3CSquareUpNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 0, 16, 16, AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize3CSquareUpLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 1, 16, 16, AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize3CSquareDownNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 2, 4, 4, AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize3CSquareDownLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 3, 4, 4, AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize3CSquareUpNearestSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 4, 10, 10, AF_INTERP_NEAREST,
                          true, &(this->subMat0));
}

TYPED_TEST(ResizeI, Resize3CSquareUpLinearSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 5, 10, 10, AF_INTERP_BILINEAR,
                          true, &(this->subMat0));
}

TYPED_TEST(ResizeI, Resize3CSquareDownNearestSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 6, 3, 3, AF_INTERP_NEAREST,
                          true, &(this->subMat0));
}

TYPED_TEST(ResizeI, Resize3CSquareDownLinearSubref)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/square.test"), 8, 3, 3, AF_INTERP_BILINEAR,
                          true, &(this->subMat1));
}

TYPED_TEST(Resize, Resize1CLargeUpNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 0, 256, 256, AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CLargeUpLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 1, 256, 256, AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize1CLargeDownNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 2, 32, 32, AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CLargeDownLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 3, 32, 32, AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize1CLargeUpNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 0, 256, 256, AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize1CLargeUpLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 1, 256, 256, AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize1CLargeDownNearest)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 2, 32, 32, AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize1CLargeDownLinear)
{
    resizeTest<TypeParam>(string(TEST_DIR"/resize/large.test"), 3, 32, 32, AF_INTERP_BILINEAR);
}

template<typename T>
void resizeArgsTest(af_err err, string pTestFile, const af::dim4 odims, const af_interp_type method)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T>>   in;
    vector<vector<T>>   tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    af::dim4 dims = numDims[0];

    af_array inArray = 0;
    af_array outArray = 0;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(err, af_resize(&outArray, inArray, odims[0], odims[1], method));

    if(inArray != 0) af_destroy_array(inArray);
    if(outArray != 0) af_destroy_array(outArray);
}

TYPED_TEST(Resize,InvalidArgsDims0)
{
    af::dim4 dims(0, 5, 2, 1);
    resizeArgsTest<TypeParam>(AF_ERR_SIZE, string(TEST_DIR"/resize/square.test"), dims, AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize,InvalidArgsMethod)
{
    af::dim4 dims(10, 10, 1, 1);
    resizeArgsTest<TypeParam>(AF_ERR_ARG, string(TEST_DIR"/resize/square.test"), dims, AF_INTERP_CUBIC);
}

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Resize, CPP)
{
    if (noDoubleTests<float>()) return;

    vector<af::dim4> numDims;
    vector<vector<float>>   in;
    vector<vector<float>>   tests;
    readTests<float, float, float>(string(TEST_DIR"/resize/square.test"),numDims,in,tests);

    af::dim4 dims = numDims[0];
    af::array input(dims, &(in[0].front()));
    af::array output = af::resize(input, 16, 16);

    // Get result
    af::dim4 odims(16, 16, dims[2], dims[3]);
    float* outData = new float[odims.elements()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[0].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(tests[0][elIter], outData[elIter], 0.0001) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;
}
