/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
class GaussianKernel : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float> TestTypes;

// register the type list
TYPED_TEST_CASE(GaussianKernel, TestTypes);

template<typename T>
void gaussianKernelTest(string pFileName, double sigma)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4>     numDims;
    vector<vector<int> > in;
    vector<vector<T> >   tests;

    readTestsFromFile<int,T>(pFileName, numDims, in, tests);

    af_array outArray  = 0;

    vector<int> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_gaussian_kernel(&outArray, input[0], input[1], sigma, sigma));

    dim_t outElems = 0;
    ASSERT_EQ(AF_SUCCESS, af_get_elements(&outElems, outArray));
    T *outData = new T[outElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<T> currGoldBar(tests[0].begin(), tests[0].end());
    size_t nElems = currGoldBar.size();

    ASSERT_EQ(outElems, (dim_t)nElems);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(GaussianKernel, Small1D)
{
    gaussianKernelTest<TypeParam>(string(TEST_DIR"/gaussian/gauss1_7.test"), 0.0);
}

TYPED_TEST(GaussianKernel, Large1D)
{
    gaussianKernelTest<TypeParam>(string(TEST_DIR"/gaussian/gauss1_15.test"), 0.0);
}

TYPED_TEST(GaussianKernel, Small1DWithSigma)
{
    gaussianKernelTest<TypeParam>(string(TEST_DIR"/gaussian/gauss1_7_sigma1.test"), 1.0);
}

TYPED_TEST(GaussianKernel, SmallSmall2D)
{
    gaussianKernelTest<TypeParam>(string(TEST_DIR"/gaussian/gauss2_7x7.test"), 0.0);
}

TYPED_TEST(GaussianKernel, LargeSmall2D)
{
    gaussianKernelTest<TypeParam>(string(TEST_DIR"/gaussian/gauss2_15x7.test"), 0.0);
}

TYPED_TEST(GaussianKernel, LargeLarge2D)
{
    gaussianKernelTest<TypeParam>(string(TEST_DIR"/gaussian/gauss2_15x15.test"), 0.0);
}

TYPED_TEST(GaussianKernel, SmallSmall2DWithSigma)
{
    gaussianKernelTest<TypeParam>(string(TEST_DIR"/gaussian/gauss2_7x7_sigma1.test"), 1.0);
}

//////////////////////////////// CPP ////////////////////////////////////
// test mean_all interface using cpp api

#include <iostream>

void gaussianKernelTestCPP(string pFileName, double sigma)
{
    using af::array;
    using af::gaussianKernel;

    vector<af::dim4>       numDims;
    vector<vector<int> >   in;
    vector<vector<float> > tests;

    readTestsFromFile<int,float>(pFileName, numDims, in, tests);

    vector<int> input(in[0].begin(), in[0].end());

    array out = gaussianKernel(input[0], input[1], sigma, sigma);

    dim_t outElems = out.elements();
    float *outData = new float[outElems];
    out.host(outData);

    vector<float> currGoldBar(tests[0].begin(), tests[0].end());
    size_t nElems = currGoldBar.size();

    ASSERT_EQ(outElems, (dim_t)nElems);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(GaussianKernel, Small1D_CPP)
{
    gaussianKernelTestCPP(string(TEST_DIR"/gaussian/gauss1_7.test"), 0.0);
}

TEST(GaussianKernel, Small1DWithSigma_CPP)
{
    gaussianKernelTestCPP(string(TEST_DIR"/gaussian/gauss1_7_sigma1.test"), 1.0);
}

TEST(GaussianKernel, SmallSmall2D_CPP)
{
    gaussianKernelTestCPP(string(TEST_DIR"/gaussian/gauss2_7x7.test"), 0.0);
}

TEST(GaussianKernel, SmallSmall2DWithSigma_CPP)
{
    gaussianKernelTestCPP(string(TEST_DIR"/gaussian/gauss2_7x7_sigma1.test"), 1.0);
}
