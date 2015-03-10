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

using std::string;
using std::vector;

template<typename T>
class MedianFilter : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(MedianFilter, TestTypes);

template<typename T>
void medfiltTest(string pTestFile, dim_type w_len, dim_type w_wid, af_pad_type pad)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4>  numDims;
    vector<vector<T> >      in;
    vector<vector<T> >   tests;

    readTests<T,T,int>(pTestFile, numDims, in, tests);

    af::dim4 dims      = numDims[0];
    af_array outArray  = 0;
    af_array inArray   = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_medfilt(&outArray, inArray, w_len, w_wid, pad));

    T *outData = new T[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<T> currGoldBar = tests[0];
    size_t nElems        = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

TYPED_TEST(MedianFilter, ZERO_PAD_3x3)
{
    medfiltTest<TypeParam>(string(TEST_DIR"/medianfilter/zero_pad_3x3_window.test"), 3, 3, AF_ZERO);
}

TYPED_TEST(MedianFilter, SYMMETRIC_PAD_3x3)
{
    medfiltTest<TypeParam>(string(TEST_DIR"/medianfilter/symmetric_pad_3x3_window.test"), 3, 3, AF_SYMMETRIC);
}

TYPED_TEST(MedianFilter, BATCH_ZERO_PAD_3x3)
{
    medfiltTest<TypeParam>(string(TEST_DIR"/medianfilter/batch_zero_pad_3x3_window.test"), 3, 3, AF_ZERO);
}

TYPED_TEST(MedianFilter, BATCH_SYMMETRIC_PAD_3x3)
{
    medfiltTest<TypeParam>(string(TEST_DIR"/medianfilter/batch_symmetric_pad_3x3_window.test"), 3, 3, AF_SYMMETRIC);
}

template<typename T,bool isColor>
void medfiltImageTest(string pTestFile, dim_type w_len, dim_type w_wid)
{
    if (noDoubleTests<T>()) return;

    using af::dim4;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_type> outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {

        af_array inArray  = 0;
        af_array outArray = 0;
        af_array goldArray= 0;
        dim_type nElems   = 0;

        inFiles[testId].insert(0,string(TEST_DIR"/medianfilter/"));
        outFiles[testId].insert(0,string(TEST_DIR"/medianfilter/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_load_image(&goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        ASSERT_EQ(AF_SUCCESS, af_medfilt(&outArray, inArray, w_len, w_wid, AF_ZERO));

        T * outData = new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        T * goldData= new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData, goldArray));

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData, outData, 0.018f));

        ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_destroy_array(goldArray));
    }
}

template<typename T>
void medfiltInputTest(void)
{
    if (noDoubleTests<T>()) return;

    af_array inArray   = 0;
    af_array outArray  = 0;

    vector<T>   in(100, 1);

    // Check for 4D inputs
    af::dim4 dims(5, 5, 2, 2);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_medfilt(&outArray, inArray, 1, 1, AF_ZERO));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));

    // Check for 1D inputs
    dims = af::dim4(100, 1, 1, 1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_medfilt(&outArray, inArray, 1, 1, AF_ZERO));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
}

TYPED_TEST(MedianFilter, InvalidArray)
{
    medfiltInputTest<TypeParam>();
}

template<typename T>
void medfiltWindowTest(void)
{
    if (noDoubleTests<T>()) return;

    af_array inArray   = 0;
    af_array outArray  = 0;

    vector<T>   in(100, 1);

    // Check for 4D inputs
    af::dim4 dims(10, 10, 1, 1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_medfilt(&outArray, inArray, -1, -1, AF_ZERO));

    ASSERT_EQ(AF_ERR_ARG, af_medfilt(&outArray, inArray, 3, 5, AF_ZERO));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
}

TYPED_TEST(MedianFilter, InvalidWindow)
{
    medfiltWindowTest<TypeParam>();
}

template<typename T>
void medfiltPadTest(void)
{
    if (noDoubleTests<T>()) return;

    af_array inArray   = 0;
    af_array outArray  = 0;

    vector<T>   in(100, 1);

    // Check for 4D inputs
    af::dim4 dims(10, 10, 1, 1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_medfilt(&outArray, inArray, 3, 3, af_pad_type(3)));

    ASSERT_EQ(AF_ERR_ARG, af_medfilt(&outArray, inArray, 3, 3, af_pad_type(-1)));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
}

TYPED_TEST(MedianFilter, InvalidPadType)
{
    medfiltPadTest<TypeParam>();
}


//////////////////////////////////// CPP ////////////////////////////////////
//
TEST(MedianFilter, CPP)
{
    if (noDoubleTests<float>()) return;

    const dim_type w_len = 3;
    const dim_type w_wid = 3;

    vector<af::dim4>  numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float,float,int>(string(TEST_DIR"/medianfilter/batch_symmetric_pad_3x3_window.test"),
                               numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af::array input(dims, &(in[0].front()));
    af::array output = af::medfilt(input, w_len, w_wid, AF_SYMMETRIC);

    float *outData = new float[dims.elements()];
    output.host((void*)outData);

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
}
