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
using af::cfloat;
using af::cdouble;

template<typename T>
class Convolve : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<cdouble, cfloat, float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Convolve, TestTypes);

template<typename T>
void convolveTest(string pTestFile, int baseDim, bool expand)
{
    if (noDoubleTests<T>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<T> >      in;
    vector<vector<T> >   tests;

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

    af_conv_mode mode = expand ? AF_CONV_EXPAND : AF_CONV_DEFAULT;
    switch(baseDim) {
    case 1: ASSERT_EQ(AF_SUCCESS, af_convolve1(&outArray, signal, filter, mode, AF_CONV_AUTO)); break;
    case 2: ASSERT_EQ(AF_SUCCESS, af_convolve2(&outArray, signal, filter, mode, AF_CONV_AUTO)); break;
    case 3: ASSERT_EQ(AF_SUCCESS, af_convolve3(&outArray, signal, filter, mode, AF_CONV_AUTO)); break;
    }

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    T *outData            = new T[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_release_array(filter));
}

TYPED_TEST(Convolve, Vector)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector.test"), 1, true);
}

TYPED_TEST(Convolve, Rectangle)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle.test"), 2, true);
}

TYPED_TEST(Convolve, Cuboid)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid.test"), 3, true);
}

TYPED_TEST(Convolve, Vector_Many2One)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector_many2one.test"), 1, true);
}

TYPED_TEST(Convolve, Rectangle_Many2One)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle_many2one.test"), 2, true);
}

TYPED_TEST(Convolve, Cuboid_Many2One)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid_many2one.test"), 3, true);
}

TYPED_TEST(Convolve, Vector_Many2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector_many2many.test"), 1, true);
}

TYPED_TEST(Convolve, Rectangle_Many2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle_many2many.test"), 2, true);
}

TYPED_TEST(Convolve, Cuboid_Many2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid_many2many.test"), 3, true);
}

TYPED_TEST(Convolve, Vector_One2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector_one2many.test"), 1, true);
}

TYPED_TEST(Convolve, Rectangle_One2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle_one2many.test"), 2, true);
}

TYPED_TEST(Convolve, Cuboid_One2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid_one2many.test"), 3, true);
}

TYPED_TEST(Convolve, Same_Vector)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector_same.test"), 1, false);
}

TYPED_TEST(Convolve, Same_Rectangle)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle_same.test"), 2, false);
}

TYPED_TEST(Convolve, Same_Cuboid)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid_same.test"), 3, false);
}

TYPED_TEST(Convolve, Same_Vector_Many2One)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector_same_many2one.test"), 1, false);
}

TYPED_TEST(Convolve, Same_Rectangle_Many2One)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle_same_many2one.test"), 2, false);
}

TYPED_TEST(Convolve, Same_Cuboid_Many2One)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid_same_many2one.test"), 3, false);
}

TYPED_TEST(Convolve, Same_Vector_Many2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector_same_many2many.test"), 1, false);
}

TYPED_TEST(Convolve, Same_Rectangle_Many2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle_same_many2many.test"), 2, false);
}

TYPED_TEST(Convolve, Same_Cuboid_Many2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid_same_many2many.test"), 3, false);
}

TYPED_TEST(Convolve, Same_Vector_One2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/vector_same_one2many.test"), 1, false);
}

TYPED_TEST(Convolve, Same_Rectangle_One2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/rectangle_same_one2many.test"), 2, false);
}

TYPED_TEST(Convolve, Same_Cuboid_One2Many)
{
    convolveTest<TypeParam>(string(TEST_DIR"/convolve/cuboid_same_one2many.test"), 3, false);
}

template<typename T>
void sepConvolveTest(string pTestFile, bool expand)
{
    if (noDoubleTests<T>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<T> >      in;
    vector<vector<T> >   tests;

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

    af_conv_mode  mode = expand ? AF_CONV_EXPAND : AF_CONV_DEFAULT;
    ASSERT_EQ(AF_SUCCESS, af_convolve2_sep(&outArray, c_filter, r_filter, signal, mode));

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    T *outData            = new T[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_release_array(c_filter));
    ASSERT_EQ(AF_SUCCESS, af_release_array(r_filter));
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
    if (noDoubleTests<float>()) return;
    if (noDoubleTests<int>()) return;
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

    ASSERT_EQ(AF_ERR_ARG, af_convolve2_sep(&outArray, c_filter, r_filter, signal, AF_CONV_EXPAND));

    ASSERT_EQ(AF_SUCCESS, af_release_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_release_array(c_filter));
    ASSERT_EQ(AF_SUCCESS, af_release_array(r_filter));
}

TEST(Convolve, Separable_DimCheck)
{
    if (noDoubleTests<float>()) return;
    if (noDoubleTests<int>()) return;

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

    ASSERT_EQ(AF_ERR_ARG, af_convolve2_sep(&outArray, c_filter, r_filter, signal, AF_CONV_EXPAND));

    ASSERT_EQ(AF_SUCCESS, af_release_array(c_filter));
    ASSERT_EQ(AF_SUCCESS, af_release_array(r_filter));
    ASSERT_EQ(AF_SUCCESS, af_release_array(signal));
}

TEST(Convolve1, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/vector_same.test"), numDims, in, tests);

    //![ex_image_convolve1]
    //vector<dim4> numDims;
    //vector<vector<float> > in;
    af::array signal(numDims[0], &(in[0].front()));
    //signal dims = [32 1 1 1]
    af::array filter(numDims[1], &(in[1].front()));
    //filter dims = [4 1 1 1]

    af::array output = convolve1(signal, filter, AF_CONV_DEFAULT);
    //output dims = [32 1 1 1] - same as input since expand(3rd argument is false)
    //None of the dimensions > 1 has lenght > 1, so no batch mode is activated.
    //![ex_image_convolve1]

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];
    output.host(outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(Convolve2, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/rectangle_same_one2many.test"), numDims, in, tests);

    //![ex_image_convolve2]
    //vector<dim4> numDims;
    //vector<vector<float> > in;
    af::array signal(numDims[0], &(in[0].front()));
    //signal dims = [15 17 1 1]
    af::array filter(numDims[1], &(in[1].front()));
    //filter dims = [5 5 2 1]

    af::array output = convolve2(signal, filter, AF_CONV_DEFAULT);
    //output dims = [15 17 1 1] - same as input since expand(3rd argument is false)
    //however, notice that the 3rd dimension of filter is > 1.
    //So, one to many batch mode will be activated automatically
    //where the 2d input signal is convolved with each 2d filter
    //and the result will written corresponding slice in the output 3d array
    //![ex_image_convolve2]

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];
    output.host(outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(Convolve3, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/cuboid_same_many2many.test"), numDims, in, tests);

    //![ex_image_convolve3]
    //vector<dim4> numDims;
    //vector<vector<float> > in;
    af::array signal(numDims[0], &(in[0].front()));
    //signal dims = [10 11 2 2]
    af::array filter(numDims[1], &(in[1].front()));
    //filter dims = [4 2 3 2]

    af::array output = convolve3(signal, filter, AF_CONV_DEFAULT);
    //output dims = [10 11 2 2] - same as input since expand(3rd argument is false)
    //however, notice that the 4th dimension is > 1 for both signal
    //and the filter, therefore many to many batch mode will be
    //activated where each 3d signal is convolved with the corresponding 3d filter
    //![ex_image_convolve3]

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
    if (noDoubleTests<float>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/separable_conv2d_same_rectangle_batch.test"),
                                 numDims, in, tests);

    //![ex_image_conv2_sep]
    //vector<dim4> numDims;
    //vector<vector<float> > in;
    af::array signal(numDims[0], &(in[0].front()));
    //signal dims = [3 4 2 1]
    af::array cFilter(numDims[1], &(in[1].front()));
    //coloumn filter dims = [2 1 1 1]
    af::array rFilter(numDims[2], &(in[2].front()));
    //row filter dims = [3 1 1 1]

    af::array output = convolve(cFilter, rFilter, signal, AF_CONV_DEFAULT);
    //output signal dims = [3 4 2 1] - same as input since 'expand = false'
    //notice that the input signal is 3d array, therefore
    //batch mode will be automatically activated.
    //output will be 3d array with result of each 2d array convolution(with same filter)
    //stacked along the 3rd dimension
    //![ex_image_conv2_sep]

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];

    output.host((void*)outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(Convolve, Docs_Unified_Wrapper)
{
    // This unit test doesn't necessarily need to function
    // accuracy as af::convolve is merely a wrapper to
    // af::convolve[1|2|3]
    using af::array;
    using af::dim4;
    using af::randu;
    using af::constant;
    using af::convolve;

    //![ex_image_convolve_1d]
    array a = randu(10);
    //af_print(a);
    //a [10 1 1 1] = 0.0000 0.1315 0.7556 0.4587 0.5328 0.2190 0.0470 0.6789 0.6793 0.9347
    array b = randu(4);
    //af_print(b);
    //b [4 1 1 1]  = 0.3835 0.5194 0.8310 0.0346
    array c = convolve(a, b);
    //af_print(c);
    //c [10 1 1 1] = 0.3581 0.6777 1.0750 0.7679 0.5903 0.4851 0.6598 1.2770 1.0734 0.8002
    //![ex_image_convolve_1d]

    //![ex_image_convolve_2d]
    array d = constant(0.5, 5, 5);
    //af_print(d);
    //d [5 5 1 1]
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    array e = constant(1, 2, 2);
    //af_print(e);
    //e [2 2 1 1]
    //     1.0000     1.0000
    //     1.0000     1.0000
    array f = convolve(d, e);
    //af_print(f);
    //f [5 5 1 1]
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     1.0000     1.0000     1.0000     1.0000     0.5000
    //![ex_image_convolve_2d]

    //![ex_image_convolve_3d]
    array g = constant(1, 4, 4, 4);
    //af_print(g);
    //g [4 4 4 1]
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000

    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000

    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000

    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    array h = constant(0.5, 2, 2, 2);
    //af_print(h);
    //h [2 2 2 1]
    //    0.5000     0.5000
    //    0.5000     0.5000

    //    0.5000     0.5000
    //    0.5000     0.5000

    array i = convolve(g, h);
    //af_print(i);
    //i [4 4 4 1]
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    2.0000     2.0000     2.0000     1.0000

    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    2.0000     2.0000     2.0000     1.0000

    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    2.0000     2.0000     2.0000     1.0000

    //    2.0000     2.0000     2.0000     1.0000
    //    2.0000     2.0000     2.0000     1.0000
    //    2.0000     2.0000     2.0000     1.0000
    //    1.0000     1.0000     1.0000     0.5000
    //![ex_image_convolve_3d]
}

using namespace af;

TEST(GFOR, convolve2_MO)
{
    array A = randu(5, 5, 3);
    array B = randu(5, 5, 3);
    array K = randu(3, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = convolve2(A(span, span, ii), K);
    }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = convolve2(A(span, span, ii), K);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(GFOR, convolve2_1M)
{
    array A = randu(5, 5);
    array B = randu(5, 5, 3);
    array K = randu(3, 3, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = convolve2(A, K(span, span, ii));
    }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = convolve2(A, K(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(GFOR, convolve2_MM)
{
    array A = randu(5, 5, 3);
    array B = randu(5, 5, 3);
    array K = randu(3, 3, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = convolve2(A(span, span, ii), K(span, span, ii));
    }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = convolve2(A(span, span, ii), K(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
