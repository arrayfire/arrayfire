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
#include <limits>

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
template<typename T>
class MeanFloat : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<cdouble, cfloat, float, double, int, uint, intl, uintl, char, uchar> TestTypes;
typedef ::testing::Types<cdouble, cfloat, float, double> TestTypesFloat;

// register the type list
TYPED_TEST_CASE(Mean, TestTypes);
TYPED_TEST_CASE(MeanFloat, TestTypesFloat);

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
    vector<vector<float> >        in;
    vector<vector<float> >   tests;

    readTestsFromFile<float,float>(pFileName, numDims, in, tests);

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

TYPED_TEST(MeanFloat, Dim1CubeRandomFloats)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim1_cube_random.test"), 1);
}

TYPED_TEST(Mean, Dim0HyperCube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim0_hypercube.test"), 0);
}

TYPED_TEST(MeanFloat, Dim0HyperCubeRandomFloats)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim0_hypercube_random.test"), 0);
}

TYPED_TEST(Mean, Dim2Matrix)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim2_matrix.test"), 2);
}

TYPED_TEST(Mean, Dim2Cube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim2_cube.test"), 2);
}
TYPED_TEST(MeanFloat, Dim2CubeRandomFloats)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim2_cube_random.test"), 2);
}

TYPED_TEST(Mean, Dim2HyperCube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim2_hypercube.test"), 2);
}
TYPED_TEST(Mean, Dim3HyperCube)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim3_hypercube.test"), 3);
}
TYPED_TEST(MeanFloat, Dim3HyperCubeRandomFloats)
{
    meanDimTest<TypeParam>(string(TEST_DIR"/mean/mean_dim3_hypercube_random.test"), 3);
}

template<typename T>
void testMeanOverflow()
{
    typedef typename meanOutType<T>::type outType;
    if (noDoubleTests<T>()) return;
    if (noDoubleTests<outType>()) return;

    vector<T> in(3,std::numeric_limits<T>::max());
    vector<T> tests = in;
    vector<af::dim4> test_dims;
    test_dims.push_back(af::dim4(3,1,1,1));
    test_dims.push_back(af::dim4(1,3,1,1));
    test_dims.push_back(af::dim4(1,1,3,1));
    test_dims.push_back(af::dim4(1,1,1,3));
    for(unsigned i = 0; i<test_dims.size(); i++){
        af_array outArray  = 0;
        af_array inArray   = 0;
        af::dim4 dims = test_dims[i];
        vector<T> input(in.begin(), in.end());
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()),
                                              dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_mean(&outArray, inArray, i));
        
        outType *outData = new outType[dims.elements()];

        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        vector<outType> currGoldBar(tests.begin(), tests.end());
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
}

TYPED_TEST(Mean,MeanComputationOverflow)
{
    testMeanOverflow<TypeParam>();
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
