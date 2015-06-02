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
typedef ::testing::Types<cdouble, cfloat, float, double, int, uint, intl, uintl, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Mean, TestTypes);

template<typename T>
struct f32HelperType {
   typedef typename cond_type<is_same_type<T, double>::value,
                                             double,
                                             float>::type type;
};

template<typename T>
struct c32HelperType {
   typedef typename cond_type<is_same_type<T, cfloat>::value,
                                             cfloat,
                                             typename f32HelperType<T>::type >::type type;
};

template<typename T>
struct elseType {
   typedef typename cond_type< is_same_type<T, uintl>::value ||
                               is_same_type<T, intl>::value,
                                              double,
                                              T>::type type;
};

template<typename T>
struct meanOutType {
   typedef typename cond_type< is_same_type<T, float>::value ||
                               is_same_type<T, int>::value ||
                               is_same_type<T, uint>::value ||
                               is_same_type<T, uchar>::value ||
                               is_same_type<T, char>::value,
                                              float,
                              typename elseType<T>::type>::type type;
};

template<typename T>
void meanDimTest(string pFileName, dim_t dim)
{
    typedef typename meanOutType<T>::type outType;
    if (noDoubleTests<T>()) return;
    if (noDoubleTests<outType>()) return;

    vector<af::dim4>      numDims;
    vector<vector<int> >        in;
    vector<vector<float> >   tests;

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
        ASSERT_NEAR(::real(currGoldBar[elIter]), ::real(outData[elIter]), 1.0e-3)<< "at: " << elIter<< std::endl;
        ASSERT_NEAR(::imag(currGoldBar[elIter]), ::imag(outData[elIter]), 1.0e-3)<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(Mean, Dim0Matrix)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim0_matrix.test"), 0);
}

TYPED_TEST(Mean, Dim1Cube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim1_cube.test"), 1);
}

TYPED_TEST(Mean, Dim0HyperCube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim0_hypercube.test"), 0);
}

TYPED_TEST(Mean, Dim2Matrix)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim2_matrix.test"), 2);
}

TYPED_TEST(Mean, Dim2Cube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim2_cube.test"), 2);
}

TYPED_TEST(Mean, Dim2HyperCube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim2_hypercube.test"), 2);
}

//////////////////////////////// CPP ////////////////////////////////////
// test mean_all interface using cpp api

#include <iostream>

template<typename T>
void testCPPMean(T const_value, af::dim4 dims)
{
    typedef typename meanOutType<T>::type outType;
    if (noDoubleTests<T>()) return;
    if (noDoubleTests<outType>()) return;

    using af::array;
    using af::mean;

    vector<T> hundred(dims.elements(), const_value);

    outType gold = outType(0);
    //for(auto i:hundred) gold += i;
    for(int i = 0; i < (int)hundred.size(); i++) {
        gold = gold + hundred[i];
    }
    gold = gold / dims.elements();

    array a(dims, &(hundred.front()));
    outType output = mean<outType>(a);

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);
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
