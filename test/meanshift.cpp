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
#include <string>
#include <vector>
#include <testHelpers.hpp>
#include <cmath>

using std::string;
using std::vector;
using af::dim4;

template<typename T>
class Meanshift : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;

TYPED_TEST_CASE(Meanshift, TestTypes);

TYPED_TEST(Meanshift, InvalidArgs)
{
    if (noDoubleTests<TypeParam>()) return;

    vector<TypeParam>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims = af::dim4(100,1,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<TypeParam>::af_type));
    ASSERT_EQ(AF_ERR_SIZE, af_mean_shift(&outArray, inArray, 0.12f, 0.34f, 5, true));
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

template<typename T, bool isColor>
void meanshiftTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t> outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {

        af_array inArray     = 0;
        af_array inArray_f32 = 0;
        af_array outArray    = 0;
        af_array goldArray   = 0;
        dim_t nElems      = 0;

        inFiles[testId].insert(0,string(TEST_DIR"/meanshift/"));
        outFiles[testId].insert(0,string(TEST_DIR"/meanshift/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray_f32, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, conv_image<T>(&inArray, inArray_f32));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        ASSERT_EQ(AF_SUCCESS, af_mean_shift(&outArray, inArray, 2.25f, 25.56f, 5, isColor));

        T * outData = new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        T * goldData= new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData, goldArray));

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData, outData, 0.07f));

        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray_f32));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(goldArray));
    }
}

// create a list of types to be tested
// FIXME: since af_load_image returns only f32 type arrays
//       only float, double data types test are enabled & passing
//       Note: compareArraysRMSD is handling upcasting while working
//       with two different type of types
//
#define IMAGE_TESTS(T)                                                      \
    TEST(Meanshift, Grayscale_##T)                                          \
    {                                                                       \
        meanshiftTest<T, false>(string(TEST_DIR"/meanshift/gray.test"));    \
    }                                                                       \
    TEST(Meanshift, Color_##T)                                              \
    {                                                                       \
        meanshiftTest<T, true>(string(TEST_DIR"/meanshift/color.test"));    \
    }

IMAGE_TESTS(float )
IMAGE_TESTS(double)


//////////////////////////////////////// CPP ///////////////////////////////
//
TEST(Meanshift, Color_CPP)
{
    if (noDoubleTests<float>()) return;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t> outSizes;
    vector<string>   outFiles;

    readImageTests(string(TEST_DIR"/meanshift/color.test"), inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {
        inFiles[testId].insert(0,string(TEST_DIR"/meanshift/"));
        outFiles[testId].insert(0,string(TEST_DIR"/meanshift/"));

        af::array img   = af::loadImage(inFiles[testId].c_str(), true);
        af::array gold  = af::loadImage(outFiles[testId].c_str(), true);
        dim_t nElems = gold.elements();
        af::array output= af::meanShift(img, 2.25f, 25.56f, 5, true);

        float * outData = new float[nElems];
        output.host((void*)outData);

        float * goldData= new float[nElems];
        gold.host((void*)goldData);

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData, outData, 0.07f));
        // cleanup
        delete[] outData;
        delete[] goldData;
    }
}

TEST(meanshift, GFOR)
{
    using namespace af;

    dim4 dims = dim4(10, 10, 3);
    array A = iota(dims);
    array B = constant(0, dims);

    gfor(seq ii, 3) {
        B(span, span, ii) = meanShift(A(span, span, ii), 3, 5, 3);
    }

    for(int ii = 0; ii < 3; ii++) {
        array c_ii = meanShift(A(span, span, ii), 3, 5, 3);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
