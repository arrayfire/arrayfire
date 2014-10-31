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

using std::vector;
using std::string;
using af::af_cfloat;
using af::af_cdouble;

template<typename T>
class Convolve : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<af_cdouble, af_cfloat, float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Convolve, TestTypes);

template<typename T, int baseDim>
void convolveTest(string pTestFile, bool expand)
{
    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<T>>      in;
    vector<vector<T>>   tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    dim4 fDims        = numDims[1];
    af_array signal   = 0;
    af_array filter   = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&signal, &(in[0].front()),
                sDims.ndims(), sDims.get(), (af_dtype)af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&filter, &(in[1].front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    switch(baseDim) {
        case 1: ASSERT_EQ(AF_SUCCESS, af_convolve1(&outArray, signal, filter, expand)); break;
        case 2: ASSERT_EQ(AF_SUCCESS, af_convolve2(&outArray, signal, filter, expand)); break;
        case 3: ASSERT_EQ(AF_SUCCESS, af_convolve3(&outArray, signal, filter, expand)); break;
    }

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    T *outData            = new T[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(filter));
}

TYPED_TEST(Convolve, Vector)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector.test"), true);
}

TYPED_TEST(Convolve, Rectangle)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle.test"), true);
}

TYPED_TEST(Convolve, Cuboid)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid.test"), true);
}

TYPED_TEST(Convolve, Vector_Many2One)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_many2one.test"), true);
}

TYPED_TEST(Convolve, Rectangle_Many2One)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_many2one.test"), true);
}

TYPED_TEST(Convolve, Cuboid_Many2One)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_many2one.test"), true);
}

TYPED_TEST(Convolve, Vector_Many2Many)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_many2many.test"), true);
}

TYPED_TEST(Convolve, Rectangle_Many2Many)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_many2many.test"), true);
}

TYPED_TEST(Convolve, Cuboid_Many2Many)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_many2many.test"), true);
}

TYPED_TEST(Convolve, Vector_One2Many)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_one2many.test"), true);
}

TYPED_TEST(Convolve, Rectangle_One2Many)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_one2many.test"), true);
}

TYPED_TEST(Convolve, Cuboid_One2Many)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_one2many.test"), true);
}

TYPED_TEST(Convolve, Same_Vector)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same.test"), false);
}

TYPED_TEST(Convolve, Same_Rectangle)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same.test"), false);
}

TYPED_TEST(Convolve, Same_Cuboid)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same.test"), false);
}

TYPED_TEST(Convolve, Same_Vector_Many2One)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same_many2one.test"), false);
}

TYPED_TEST(Convolve, Same_Rectangle_Many2One)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same_many2one.test"), false);
}

TYPED_TEST(Convolve, Same_Cuboid_Many2One)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same_many2one.test"), false);
}

TYPED_TEST(Convolve, Same_Vector_Many2Many)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same_many2many.test"), false);
}

TYPED_TEST(Convolve, Same_Rectangle_Many2Many)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same_many2many.test"), false);
}

TYPED_TEST(Convolve, Same_Cuboid_Many2Many)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same_many2many.test"), false);
}

TYPED_TEST(Convolve, Same_Vector_One2Many)
{
    convolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same_one2many.test"), false);
}

TYPED_TEST(Convolve, Same_Rectangle_One2Many)
{
    convolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same_one2many.test"), false);
}

TYPED_TEST(Convolve, Same_Cuboid_One2Many)
{
    convolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same_one2many.test"), false);
}

TEST(Convolve, TypeCheck)
{
    using af::dim4;

    dim4 sDims(10, 1, 1, 1);
    dim4 fDims(4, 1, 1, 1);

    vector<float> in(10,1);
    vector<int>   filt(4,1);

    af_array signal   = 0;
    af_array filter   = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&signal, &(in.front()),
                sDims.ndims(), sDims.get(), (af_dtype)af::dtype_traits<float>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&filter, &(filt.front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<int>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_convolve1(&outArray, signal, filter, true));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(filter));
}

TEST(Convolve, DimCheck)
{
    using af::dim4;

    dim4 sDims(10, 1, 1, 1);
    dim4 fDims(4, 1, 1, 1);

    vector<float> in(10,1);
    vector<int>   filt(4,1);

    af_array signal   = 0;
    af_array filter   = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&signal, &(in.front()),
                sDims.ndims(), sDims.get(), (af_dtype)af::dtype_traits<float>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&filter, &(filt.front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<int>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_convolve2(&outArray, signal, filter, true));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(filter));

    fDims[0] = fDims[2] = 2;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&filter, &(filt.front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<int>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_convolve1(&outArray, signal, filter, true));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(filter));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(signal));
}

template<typename T>
void sepConvolveTest(string pTestFile, bool expand)
{
    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<T>>      in;
    vector<vector<T>>   tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    dim4 cfDims       = numDims[1];
    dim4 rfDims       = numDims[2];
    af_array signal   = 0;
    af_array c_filter = 0;
    af_array r_filter = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&signal, &(in[0].front()),
                sDims.ndims(), sDims.get(), (af_dtype)af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&c_filter, &(in[1].front()),
                cfDims.ndims(), cfDims.get(), (af_dtype)af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&r_filter, &(in[2].front()),
                rfDims.ndims(), rfDims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_convolve2_sep(&outArray, signal, c_filter, r_filter, expand));

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    T *outData            = new T[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(c_filter));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(r_filter));
}

TYPED_TEST(Convolve, Separable2D_Full)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_full.test"), true);
}

TYPED_TEST(Convolve, Separable2D_Full_Batch)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_full_batch.test"), true);
}

TYPED_TEST(Convolve, Separable2D_Full_Rectangle)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_full_rectangle.test"), true);
}

TYPED_TEST(Convolve, Separable2D_Full_Rectangle_Batch)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_full_rectangle_batch.test"), true);
}

TYPED_TEST(Convolve, Separable2D_Same)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_same.test"), false);
}

TYPED_TEST(Convolve, Separable2D_Same_Batch)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_same_batch.test"), false);
}

TYPED_TEST(Convolve, Separable2D_Same_Rectangle)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_same_rectangle.test"), false);
}

TYPED_TEST(Convolve, Separable2D_Same_Rectangle_Batch)
{
    sepConvolveTest<TypeParam>(string(TEST_DIR"/convolve/separable_conv2d_same_rectangle_batch.test"), false);
}

TEST(Convolve, Separable_TypeCheck)
{
    using af::dim4;

    dim4 sDims(10, 1, 1, 1);
    dim4 fDims(4, 1, 1, 1);

    vector<float> in(10,1);
    vector<int>   filt(4,1);

    af_array signal   = 0;
    af_array c_filter = 0;
    af_array r_filter = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&signal, &(in.front()),
                sDims.ndims(), sDims.get(), (af_dtype)af::dtype_traits<float>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&c_filter, &(filt.front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<int>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&r_filter, &(filt.front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<int>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_convolve2_sep(&outArray, signal, c_filter, r_filter, true));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(c_filter));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(r_filter));
}

TEST(Convolve, Separable_DimCheck)
{
    using af::dim4;

    dim4 sDims(10, 1, 1, 1);
    dim4 fDims(4, 1, 1, 1);

    vector<float> in(10,1);
    vector<int>   filt(4,1);

    af_array signal   = 0;
    af_array c_filter = 0;
    af_array r_filter = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&signal, &(in.front()),
                sDims.ndims(), sDims.get(), (af_dtype)af::dtype_traits<float>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&c_filter, &(filt.front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<int>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&r_filter, &(filt.front()),
                fDims.ndims(), fDims.get(), (af_dtype)af::dtype_traits<int>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_convolve2_sep(&outArray, signal, c_filter, r_filter, true));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(c_filter));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(r_filter));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(signal));
}


TEST(Convolve, CPP)
{
    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float>>      in;
    vector<vector<float>>   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/cuboid_same_many2many.test"), numDims, in, tests);

    af::array signal(numDims[0], &(in[0].front()));
    af::array filter(numDims[1], &(in[1].front()));

    af::array output = convolve3(signal, filter, false);

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];
    output.host(outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(Convolve, separable_CPP)
{
    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float>>      in;
    vector<vector<float>>   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/separable_conv2d_same_rectangle_batch.test"),
                                 numDims, in, tests);

    af::array signal(numDims[0], &(in[0].front()));
    af::array cFilter(numDims[1], &(in[1].front()));
    af::array rFilter(numDims[2], &(in[2].front()));

    af::array output = convolve2(signal, cFilter, rFilter, false);

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];

    output.host((void*)outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}
