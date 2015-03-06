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
using af::cdouble;
using af::cfloat;

template<typename T>
class Mean : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<cdouble, cfloat, float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Mean, TestTypes);

template<typename T>
using f32HelperType = typename std::conditional<std::is_same<T, double>::value,
                                             double,
                                             float>::type;

template<typename T>
using c32HelperType = typename std::conditional<std::is_same<T, cfloat>::value,
                                             cfloat,
                                             f32HelperType<T>>::type;
template<typename T>
using meanOutType = typename std::conditional<std::is_same<T, cdouble>::value,
                                              cdouble,
                                              c32HelperType<T>>::type;

template<typename T, dim_type dim>
void meanDimTest(string pFileName)
{
    if (noDoubleTests<T>()) return;

    typedef meanOutType<T> outType;
    if (noDoubleTests<T>()) return;

    vector<af::dim4>      numDims;
    vector<vector<int>>        in;
    vector<vector<float>>   tests;

    readTestsFromFile<int,float>(pFileName, numDims, in, tests);

    af::dim4 dims      = numDims[0];
    af_array outArray  = 0;
    af_array inArray   = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_mean(&outArray, inArray, dim));

    outType *outData = new outType[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<outType> currGoldBar(tests[0].begin(), tests[0].end());
    size_t nElems = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(std::real(currGoldBar[elIter]), std::real(outData[elIter]), 1.0e-3)<< "at: " << elIter<< std::endl;
        ASSERT_NEAR(std::imag(currGoldBar[elIter]), std::imag(outData[elIter]), 1.0e-3)<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

TYPED_TEST(Mean, Dim0Matrix)
{
    meanDimTest<TypeParam, 0>(string(TEST_DIR"/mean/mean_dim0_matrix.test"));
}

TYPED_TEST(Mean, Dim1Cube)
{
    meanDimTest<TypeParam, 1>(string(TEST_DIR"/mean/mean_dim1_cube.test"));
}

TYPED_TEST(Mean, Dim0HyperCube)
{
    meanDimTest<TypeParam, 0>(string(TEST_DIR"/mean/mean_dim0_hypercube.test"));
}

TYPED_TEST(Mean, Dim2Matrix)
{
    meanDimTest<TypeParam, 2>(string(TEST_DIR"/mean/mean_dim2_matrix.test"));
}

TYPED_TEST(Mean, Dim2Cube)
{
    meanDimTest<TypeParam, 2>(string(TEST_DIR"/mean/mean_dim2_cube.test"));
}

TYPED_TEST(Mean, Dim2HyperCube)
{
    meanDimTest<TypeParam, 2>(string(TEST_DIR"/mean/mean_dim2_hypercube.test"));
}

//////////////////////////////// CPP ////////////////////////////////////
// test mean_all interface using cpp api

#include <iostream>

template<typename T>
void testCPPMean(T const_value, af::dim4 dims)
{
    if (noDoubleTests<T>()) return;

    typedef meanOutType<T> outType;

    using af::array;
    using af::mean;

    vector<T> hundred(dims.elements(), const_value);

    outType gold = outType(0);
    for(auto i:hundred) gold += i;
    gold /= dims.elements();

    array a(dims, &(hundred.front()));
    outType output = mean<outType>(a);

    ASSERT_NEAR(std::real(output), std::real(gold), 1.0e-3);
    ASSERT_NEAR(std::imag(output), std::imag(gold), 1.0e-3);
}

TEST(Mean, CPP_f64)
{
    testCPPMean<double>(2.1, af::dim4(10, 10, 1, 1));
}

TEST(Mean, CPP_f32)
{
    testCPPMean<float>(2.1f, af::dim4(10, 5, 2, 1));
}

TEST(Mean, CPP_s32)
{
    testCPPMean<int>(2, af::dim4(5, 5, 2, 2));
}

TEST(Mean, CPP_u32)
{
    testCPPMean<unsigned>(2, af::dim4(100, 1, 1, 1));
}

TEST(Mean, CPP_s8)
{
    testCPPMean<char>(2, af::dim4(5, 5, 2, 2));
}

TEST(Mean, CPP_u8)
{
    testCPPMean<uchar>(2, af::dim4(100, 1, 1, 1));
}

TEST(Mean, CPP_cfloat)
{
    testCPPMean<cfloat>(cfloat(2.1f), af::dim4(10, 5, 2, 1));
}

TEST(Mean, CPP_cdouble)
{
    testCPPMean<cdouble>(cdouble(2.1), af::dim4(10, 10, 1, 1));
}
