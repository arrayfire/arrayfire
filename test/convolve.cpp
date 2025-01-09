/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <cmath>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Convolve : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<cdouble, cfloat, float, double, int, uint, char, uchar,
                         short, ushort, intl, uintl>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Convolve, TestTypes);

template<typename T>
void convolveTest(string pTestFile, int baseDim, bool expand) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    dim4 fDims        = numDims[1];
    af_array signal   = 0;
    af_array filter   = 0;
    af_array outArray = 0;

    ASSERT_SUCCESS(af_create_array(&signal, &(in[0].front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&filter, &(in[1].front()), fDims.ndims(),
                                   fDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    af_conv_mode mode = expand ? AF_CONV_EXPAND : AF_CONV_DEFAULT;
    switch (baseDim) {
        case 1:
            ASSERT_SUCCESS(
                af_convolve1(&outArray, signal, filter, mode, AF_CONV_AUTO));
            break;
        case 2:
            ASSERT_SUCCESS(
                af_convolve2(&outArray, signal, filter, mode, AF_CONV_AUTO));
            break;
        case 3:
            ASSERT_SUCCESS(
                af_convolve3(&outArray, signal, filter, mode, AF_CONV_AUTO));
            break;
    }

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    vector<T> outData(nElems);

    ASSERT_SUCCESS(af_get_data_ptr((void *)&outData.front(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(af_release_array(outArray));
    ASSERT_SUCCESS(af_release_array(signal));
    ASSERT_SUCCESS(af_release_array(filter));
}

TYPED_TEST(Convolve, Vector) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/vector.test"), 1, true);
}

TYPED_TEST(Convolve, Rectangle) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/rectangle.test"), 2,
                            true);
}

TYPED_TEST(Convolve, Cuboid) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/cuboid.test"), 3, true);
}

TYPED_TEST(Convolve, Vector_Many2One) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/vector_many2one.test"),
                            1, true);
}

TYPED_TEST(Convolve, Rectangle_Many2One) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/rectangle_many2one.test"), 2, true);
}

TYPED_TEST(Convolve, Cuboid_Many2One) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/cuboid_many2one.test"),
                            3, true);
}

TYPED_TEST(Convolve, Vector_Many2Many) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/vector_many2many.test"),
                            1, true);
}

TYPED_TEST(Convolve, Rectangle_Many2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/rectangle_many2many.test"), 2, true);
}

TYPED_TEST(Convolve, Cuboid_Many2Many) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/cuboid_many2many.test"),
                            3, true);
}

TYPED_TEST(Convolve, Vector_One2Many) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/vector_one2many.test"),
                            1, true);
}

TYPED_TEST(Convolve, Rectangle_One2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/rectangle_one2many.test"), 2, true);
}

TYPED_TEST(Convolve, Cuboid_One2Many) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/cuboid_one2many.test"),
                            3, true);
}

TYPED_TEST(Convolve, Same_Vector) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/vector_same.test"), 1,
                            false);
}

TYPED_TEST(Convolve, Same_Rectangle) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/rectangle_same.test"), 2,
                            false);
}

TYPED_TEST(Convolve, Same_Cuboid) {
    convolveTest<TypeParam>(string(TEST_DIR "/convolve/cuboid_same.test"), 3,
                            false);
}

TYPED_TEST(Convolve, Same_Vector_Many2One) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/vector_same_many2one.test"), 1, false);
}

TYPED_TEST(Convolve, Same_Rectangle_Many2One) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/rectangle_same_many2one.test"), 2, false);
}

TYPED_TEST(Convolve, Same_Cuboid_Many2One) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/cuboid_same_many2one.test"), 3, false);
}

TYPED_TEST(Convolve, Same_Vector_Many2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/vector_same_many2many.test"), 1, false);
}

TYPED_TEST(Convolve, Same_Rectangle_Many2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/rectangle_same_many2many.test"), 2, false);
}

TYPED_TEST(Convolve, Same_Cuboid_Many2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/cuboid_same_many2many.test"), 3, false);
}

TYPED_TEST(Convolve, Same_Vector_One2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/vector_same_one2many.test"), 1, false);
}

TYPED_TEST(Convolve, Same_Rectangle_One2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/rectangle_same_one2many.test"), 2, false);
}

TYPED_TEST(Convolve, Same_Cuboid_One2Many) {
    convolveTest<TypeParam>(
        string(TEST_DIR "/convolve/cuboid_same_one2many.test"), 3, false);
}

template<typename T>
void sepConvolveTest(string pTestFile, bool expand) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    dim4 cfDims       = numDims[1];
    dim4 rfDims       = numDims[2];
    af_array signal   = 0;
    af_array c_filter = 0;
    af_array r_filter = 0;
    af_array outArray = 0;

    ASSERT_SUCCESS(af_create_array(&signal, &(in[0].front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&c_filter, &(in[1].front()), cfDims.ndims(),
                                   cfDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&r_filter, &(in[2].front()), rfDims.ndims(),
                                   rfDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    af_conv_mode mode = expand ? AF_CONV_EXPAND : AF_CONV_DEFAULT;
    ASSERT_SUCCESS(
        af_convolve2_sep(&outArray, c_filter, r_filter, signal, mode));

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    vector<T> outData(nElems);

    ASSERT_SUCCESS(af_get_data_ptr((void *)&outData.front(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(af_release_array(outArray));
    ASSERT_SUCCESS(af_release_array(signal));
    ASSERT_SUCCESS(af_release_array(c_filter));
    ASSERT_SUCCESS(af_release_array(r_filter));
}

TYPED_TEST(Convolve, Separable2D_Full) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_full.test"), true);
}

TYPED_TEST(Convolve, Separable2D_Full_Batch) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_full_batch.test"), true);
}

TYPED_TEST(Convolve, Separable2D_Full_Rectangle) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_full_rectangle.test"),
        true);
}

TYPED_TEST(Convolve, Separable2D_Full_Rectangle_Batch) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_full_rectangle_batch.test"),
        true);
}

TYPED_TEST(Convolve, Separable2D_Same) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_same.test"), false);
}

TYPED_TEST(Convolve, Separable2D_Same_Batch) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_same_batch.test"), false);
}

TYPED_TEST(Convolve, Separable2D_Same_Rectangle) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_same_rectangle.test"),
        false);
}

TYPED_TEST(Convolve, Separable2D_Same_Rectangle_Batch) {
    sepConvolveTest<TypeParam>(
        string(TEST_DIR "/convolve/separable_conv2d_same_rectangle_batch.test"),
        false);
}

TEST(Convolve, Separable_TypeCheck) {
    dim4 sDims(10, 1, 1, 1);
    dim4 fDims(4, 1, 1, 1);

    vector<float> in(10, 1);
    vector<int> filt(4, 1);

    af_array signal   = 0;
    af_array c_filter = 0;
    af_array r_filter = 0;
    af_array outArray = 0;

    ASSERT_SUCCESS(af_create_array(&signal, &(in.front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<float>::af_type));
    ASSERT_SUCCESS(af_create_array(&c_filter, &(filt.front()), fDims.ndims(),
                                   fDims.get(),
                                   (af_dtype)dtype_traits<int>::af_type));
    ASSERT_SUCCESS(af_create_array(&r_filter, &(filt.front()), fDims.ndims(),
                                   fDims.get(),
                                   (af_dtype)dtype_traits<int>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_convolve2_sep(&outArray, c_filter, r_filter,
                                           signal, AF_CONV_EXPAND));

    ASSERT_SUCCESS(af_release_array(signal));
    ASSERT_SUCCESS(af_release_array(c_filter));
    ASSERT_SUCCESS(af_release_array(r_filter));
}

TEST(Convolve, Separable_DimCheck) {
    dim4 sDims(10, 1, 1, 1);
    dim4 fDims(4, 1, 1, 1);

    vector<float> in(10, 1);
    vector<int> filt(4, 1);

    af_array signal   = 0;
    af_array c_filter = 0;
    af_array r_filter = 0;
    af_array outArray = 0;

    ASSERT_SUCCESS(af_create_array(&signal, &(in.front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<float>::af_type));
    ASSERT_SUCCESS(af_create_array(&c_filter, &(filt.front()), fDims.ndims(),
                                   fDims.get(),
                                   (af_dtype)dtype_traits<int>::af_type));
    ASSERT_SUCCESS(af_create_array(&r_filter, &(filt.front()), fDims.ndims(),
                                   fDims.get(),
                                   (af_dtype)dtype_traits<int>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_convolve2_sep(&outArray, c_filter, r_filter,
                                           signal, AF_CONV_EXPAND));

    ASSERT_SUCCESS(af_release_array(c_filter));
    ASSERT_SUCCESS(af_release_array(r_filter));
    ASSERT_SUCCESS(af_release_array(signal));
}

///////////////////////////////////// CPP ////////////////////////////////
//
using af::constant;
using af::max;
using af::product;
using af::randu;
using af::seq;
using af::span;
using af::sum;

TEST(Convolve1, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(string(TEST_DIR "/convolve/vector_same.test"),
                                 numDims, in, tests);

    //![ex_image_convolve1]
    // vector<dim4> numDims;
    // vector<vector<float> > in;
    array signal(numDims[0], &(in[0].front()));
    // signal dims = [32 1 1 1]
    array filter(numDims[1], &(in[1].front()));
    // filter dims = [4 1 1 1]

    array output = convolve1(signal, filter, AF_CONV_DEFAULT);
    // output dims = [32 1 1 1] - same as input since expand(3rd argument is
    // false) None of the dimensions > 1 has lenght > 1, so no batch mode is
    // activated.
    //![ex_image_convolve1]

    vector<float> currGoldBar = tests[0];
    size_t nElems             = output.elements();
    vector<float> outData(nElems);
    output.host(&outData.front());

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }
}

TEST(Convolve2, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(
        string(TEST_DIR "/convolve/rectangle_same_one2many.test"), numDims, in,
        tests);

    //![ex_image_convolve2]
    // vector<dim4> numDims;
    // vector<vector<float> > in;
    array signal(numDims[0], &(in[0].front()));
    // signal dims = [15 17 1 1]
    array filter(numDims[1], &(in[1].front()));
    // filter dims = [5 5 2 1]

    array output = convolve2(signal, filter, AF_CONV_DEFAULT);
    // output dims = [15 17 1 1] - same as input since expand(3rd argument is
    // false) however, notice that the 3rd dimension of filter is > 1. So, one
    // to many batch mode will be activated automatically where the 2d input
    // signal is convolved with each 2d filter and the result will written
    // corresponding slice in the output 3d array
    //![ex_image_convolve2]

    vector<float> currGoldBar = tests[0];
    size_t nElems             = output.elements();
    vector<float> outData(nElems);
    output.host(&outData.front());

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }
}

TEST(Convolve3, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(
        string(TEST_DIR "/convolve/cuboid_same_many2many.test"), numDims, in,
        tests);

    //![ex_image_convolve3]
    // vector<dim4> numDims;
    // vector<vector<float> > in;
    array signal(numDims[0], &(in[0].front()));
    // signal dims = [10 11 2 2]
    array filter(numDims[1], &(in[1].front()));
    // filter dims = [4 2 3 2]

    array output = convolve3(signal, filter, AF_CONV_DEFAULT);
    // output dims = [10 11 2 2] - same as input since expand(3rd argument is
    // false) however, notice that the 4th dimension is > 1 for both signal and
    // the filter, therefore many to many batch mode will be activated where
    // each 3d signal is convolved with the corresponding 3d filter
    //![ex_image_convolve3]

    vector<float> currGoldBar = tests[0];
    size_t nElems             = output.elements();
    vector<float> outData(nElems);
    output.host(&outData.front());

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }
}

TEST(Convolve, separable_CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(
        string(TEST_DIR "/convolve/separable_conv2d_same_rectangle_batch.test"),
        numDims, in, tests);

    //![ex_image_conv2_sep]
    // vector<dim4> numDims;
    // vector<vector<float> > in;
    array signal(numDims[0], &(in[0].front()));
    // signal dims = [3 4 2 1]
    array cFilter(numDims[1], &(in[1].front()));
    // coloumn filter dims = [2 1 1 1]
    array rFilter(numDims[2], &(in[2].front()));
    // row filter dims = [3 1 1 1]

    array output = convolve(cFilter, rFilter, signal, AF_CONV_DEFAULT);
    // output signal dims = [3 4 2 1] - same as input since 'expand = false'
    // notice that the input signal is 3d array, therefore
    // batch mode will be automatically activated.
    // output will be 3d array with result of each 2d array convolution(with
    // same filter) stacked along the 3rd dimension
    //![ex_image_conv2_sep]

    vector<float> currGoldBar = tests[0];
    size_t nElems             = output.elements();
    vector<float> outData(nElems);

    output.host((void *)&outData.front());

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }
}

TEST(Convolve, Docs_Unified_Wrapper) {
    // This unit test doesn't necessarily need to function
    // accuracy as convolve is merely a wrapper to
    // convolve[1|2|3]

    //![ex_image_convolve_1d]
    array a = randu(10);
    // af_print(a);
    // a [10 1 1 1] = 0.0000 0.1315 0.7556 0.4587 0.5328 0.2190 0.0470 0.6789
    // 0.6793 0.9347
    array b = randu(4);
    // af_print(b);
    // b [4 1 1 1]  = 0.3835 0.5194 0.8310 0.0346
    array c = convolve(a, b);
    // af_print(c);
    // c [10 1 1 1] = 0.3581 0.6777 1.0750 0.7679 0.5903 0.4851 0.6598
    // 1.2770 1.0734 0.8002
    //![ex_image_convolve_1d]

    //![ex_image_convolve_2d]
    array d = constant(0.5, 5, 5);
    // af_print(d);
    // d [5 5 1 1]
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    array e = constant(1, 2, 2);
    // af_print(e);
    // e [2 2 1 1]
    //     1.0000     1.0000
    //     1.0000     1.0000
    array f = convolve(d, e);
    // af_print(f);
    // f [5 5 1 1]
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     1.0000     1.0000     1.0000     1.0000     0.5000
    //![ex_image_convolve_2d]

    //![ex_image_convolve_3d]
    array g = constant(1, 4, 4, 4);
    // af_print(g);
    // g [4 4 4 1]
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
    // af_print(h);
    // h [2 2 2 1]
    //    0.5000     0.5000
    //    0.5000     0.5000

    //    0.5000     0.5000
    //    0.5000     0.5000

    array i = convolve(g, h);
    // af_print(i);
    // i [4 4 4 1]
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

TEST(GFOR, convolve2_MO) {
    array A = randu(5, 5, 3);
    array B = randu(5, 5, 3);
    array K = randu(3, 3);

    gfor(seq ii, 3) { B(span, span, ii) = convolve2(A(span, span, ii), K); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = convolve2(A(span, span, ii), K);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(GFOR, convolve2_OM) {
    array A = randu(5, 5);
    array B = randu(5, 5, 3);
    array K = randu(3, 3, 3);

    gfor(seq ii, 3) { B(span, span, ii) = convolve2(A, K(span, span, ii)); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = convolve2(A, K(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(GFOR, convolve2_MM) {
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

TEST(Convolve, 1D_C32) {
    array A = randu(10, c32);
    array B = randu(3, c32);

    array out = convolve1(A, B);
    array gld = fftConvolve1(A, B);

    cfloat acc = sum<cfloat>(out - gld);

    EXPECT_LT(std::abs(real(acc)), 1E-3);
    EXPECT_LT(std::abs(imag(acc)), 1E-3);
}

TEST(Convolve, 2D_C32) {
    array A = randu(10, 10, c32);
    array B = randu(3, 3, c32);

    array out = convolve2(A, B);
    array gld = fftConvolve2(A, B);

    cfloat acc = sum<cfloat>(out - gld);

    EXPECT_LT(std::abs(real(acc)), 1E-3);
    EXPECT_LT(std::abs(imag(acc)), 1E-3);
}

TEST(Convolve, 3D_C32) {
    array A = randu(10, 10, 3, c32);
    array B = randu(3, 3, 3, c32);

    array out = convolve3(A, B);
    array gld = fftConvolve3(A, B);

    cfloat acc = sum<cfloat>(out - gld);

    EXPECT_EQ(std::abs(real(acc)) < 1E-3, true);
    EXPECT_EQ(std::abs(imag(acc)) < 1E-3, true);
}

TEST(Convolve, 1D_C64) {
    SUPPORTED_TYPE_CHECK(double);

    array A = randu(10, c64);
    array B = randu(3, c64);

    array out = convolve1(A, B);
    array gld = fftConvolve1(A, B);

    cdouble acc = sum<cdouble>(out - gld);

    EXPECT_EQ(std::abs(real(acc)) < 1E-3, true);
    EXPECT_EQ(std::abs(imag(acc)) < 1E-3, true);
}

TEST(Convolve, 2D_C64) {
    SUPPORTED_TYPE_CHECK(double);

    array A = randu(10, 10, c64);
    array B = randu(3, 3, c64);

    array out = convolve2(A, B);
    array gld = fftConvolve2(A, B);

    cdouble acc = sum<cdouble>(out - gld);

    EXPECT_EQ(std::abs(real(acc)) < 1E-3, true);
    EXPECT_EQ(std::abs(imag(acc)) < 1E-3, true);
}

TEST(Convolve, 3D_C64) {
    SUPPORTED_TYPE_CHECK(double);

    array A = randu(10, 10, 3, c64);
    array B = randu(3, 3, 3, c64);

    array out = convolve3(A, B);
    array gld = fftConvolve3(A, B);

    cdouble acc = sum<cdouble>(out - gld);

    EXPECT_EQ(std::abs(real(acc)) < 1E-3, true);
    EXPECT_EQ(std::abs(imag(acc)) < 1E-3, true);
}

TEST(ConvolveLargeDim1D, CPP) {
    const size_t n        = 10;
    const size_t largeDim = 65535 + 1;

    float h_filter[] = {0.f, 1.f, 0.f};
    array identity_filter(3, h_filter);
    array signal = constant(1, n, 1, largeDim);

    array output  = convolve1(signal, identity_filter, AF_CONV_DEFAULT);
    array output2 = output;
    ASSERT_EQ(largeDim * n, sum<float>(output2));

    signal = constant(1, n, 1, 1, largeDim);

    output = convolve1(signal, identity_filter, AF_CONV_DEFAULT);
    ASSERT_EQ(largeDim * n, sum<float>(output));
}

TEST(ConvolveLargeDim2D, CPP) {
    const size_t n        = 10;
    const size_t largeDim = 65535 + 1;

    float h_filter[] = {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f};
    array identity_filter(3, 3, h_filter);
    array signal = constant(1, n, n, largeDim);

    array output = convolve2(signal, identity_filter, AF_CONV_DEFAULT);
    ASSERT_EQ(largeDim * n * n, sum<float>(output));

    signal = constant(1, n, n, 1, largeDim);

    output = convolve2(signal, identity_filter, AF_CONV_DEFAULT);
    ASSERT_EQ(largeDim * n * n, sum<float>(output));
}

TEST(DISABLED_ConvolveLargeDim3D, CPP) {
    const size_t n        = 3;
    const size_t largeDim = 65535 * 16 + 1;

    float h_filter[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

                        0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,

                        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    array identity_filter(3, 3, 3, h_filter);
    array signal = constant(1, n, largeDim, n);

    array output = convolve3(signal, identity_filter, AF_CONV_DEFAULT);
    ASSERT_EQ(1.f, product<float>(output));

    signal = constant(1, n, n, largeDim);

    output = convolve3(signal, identity_filter, AF_CONV_EXPAND);
    // TODO: fix product by indexing
    // ASSERT_EQ(1.f, product<float>(output));
}

TEST(Convolve, CuboidBatchLaunchBugFix) {
    std::string testFile(TEST_DIR "/convolve/conv3d_launch_bug.test");

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, float>(testFile, numDims, in, tests);

    dim4 sDims = numDims[0];
    dim4 fDims = numDims[1];

    af::array signal(sDims, in[0].data());
    af::array filter(fDims, in[1].data());

    af::array output = convolve3(signal, filter);

    ASSERT_VEC_ARRAY_NEAR(tests[0], sDims, output, 1.0e-3);
}

struct conv2_strided_params {
    string testname_;
    dim4 signal_sz_, filt_sz_, stride_, padding_, dilation_;

    conv2_strided_params(string testname, dim4 signal_sz, dim4 filt_sz,
                         dim4 stride, dim4 padding, dim4 dilation)
        : testname_(testname)
        , signal_sz_(signal_sz)
        , filt_sz_(filt_sz)
        , stride_(stride)
        , padding_(padding)
        , dilation_(dilation) {}
};

template<typename TestClass>
string testNameGenerator(
    const ::testing::TestParamInfo<typename TestClass::ParamType> info) {
    return info.param.testname_;
}

class Conv2ConsistencyTest
    : public ::testing::TestWithParam<conv2_strided_params> {};

conv2_strided_params conv2_consistency_data(dim4 signal_sz, dim4 filt_sz) {
    dim4 stride(1, 1);
    dim4 padding(filt_sz[0] / 2, filt_sz[1] / 2);
    dim4 dilation(1, 1);
    std::string testname =
        "conv2_consistency_" + std::to_string(signal_sz[0]) +
        std::to_string(signal_sz[1]) + std::to_string(signal_sz[2]) +
        std::to_string(signal_sz[3]) + "__" + std::to_string(filt_sz[0]) +
        std::to_string(filt_sz[1]) + std::to_string(filt_sz[2]) +
        std::to_string(filt_sz[3]) + "__" + "s" + std::to_string(stride[0]) +
        std::to_string(stride[1]) + "_" + "p" + std::to_string(padding[0]) +
        std::to_string(padding[1]) + "_" + "d" + std::to_string(dilation[0]) +
        std::to_string(dilation[1]);

    return conv2_strided_params(testname, signal_sz, filt_sz, stride, padding,
                                dilation);
}
vector<conv2_strided_params> genConsistencyTests() {
    // TODO: test nfilters and nfeatures
    return {conv2_consistency_data(dim4(10, 10), dim4(3, 3)),
            conv2_consistency_data(dim4(11, 11), dim4(5, 5)),
            conv2_consistency_data(dim4(12, 12), dim4(7, 7)),
            conv2_consistency_data(dim4(19, 19), dim4(9, 9)),
            conv2_consistency_data(dim4(33, 33), dim4(3, 3)),
            conv2_consistency_data(dim4(255, 255), dim4(3, 3)),
            conv2_consistency_data(dim4(256, 256), dim4(3, 3)),
            conv2_consistency_data(dim4(257, 257), dim4(3, 3))};
}

INSTANTIATE_TEST_SUITE_P(Conv2Consistency, Conv2ConsistencyTest,
                         ::testing::ValuesIn(genConsistencyTests()),
                         testNameGenerator<Conv2ConsistencyTest>);

TEST_P(Conv2ConsistencyTest, RandomConvolutions) {
    conv2_strided_params params = GetParam();
    array signal                = randn(params.signal_sz_);
    array filter                = randn(params.filt_sz_);

    array out_native = convolve2(signal, filter);
    array out = convolve2NN(signal, filter, params.stride_, params.padding_,
                            params.dilation_);

    ASSERT_ARRAYS_NEAR(out_native, out, 2e-5);
}

template<typename T>
float tolerance();

template<>
float tolerance<float>() {
    return 4e-3;
}

template<>
float tolerance<double>() {
    return 1e-4;
}

template<>
float tolerance<half_float::half>() {
    return 7e-2;
}

template<typename T>
void convolve2stridedTest(string pTestFile, dim4 stride, dim4 padding,
                          dim4 dilation) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 sDims         = numDims[0];
    dim4 fDims         = numDims[1];
    af_array signal    = 0;
    af_array filter    = 0;
    af_array convolved = 0;

    ASSERT_SUCCESS(af_create_array(&signal, &(in[0].front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&filter, &(in[1].front()), fDims.ndims(),
                                   fDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_convolve2_nn(&convolved, signal, filter, stride.ndims(),
                                   stride.get(), padding.ndims(), padding.get(),
                                   dilation.ndims(), dilation.get()));

    vector<T> &currGoldBar = tests[0];

    dim_t expectedDim0 =
        1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) /
                stride[0];
    dim_t expectedDim1 =
        1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) /
                stride[1];

    auto gdim = dim4(expectedDim0, expectedDim1, fDims[3], sDims[3]);
    ASSERT_VEC_ARRAY_NEAR(currGoldBar, gdim, convolved, tolerance<T>());

    ASSERT_SUCCESS(af_release_array(convolved));
    ASSERT_SUCCESS(af_release_array(signal));
    ASSERT_SUCCESS(af_release_array(filter));
}

template<typename T>
void convolve2GradientTest(string pTestFile, dim4 stride, dim4 padding,
                           dim4 dilation) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 sDims         = numDims[0];
    dim4 fDims         = numDims[1];
    af_array signal    = 0;
    af_array filter    = 0;
    af_array convolved = 0;

    ASSERT_SUCCESS(af_create_array(&signal, &(in[0].front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&filter, &(in[1].front()), fDims.ndims(),
                                   fDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    vector<T> &currGoldBar = tests[0];
    size_t nElems          = currGoldBar.size();

    dim_t expectedDim0 =
        1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) /
                stride[0];
    dim_t expectedDim1 =
        1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) /
                stride[1];
    dim4 cDims(expectedDim0, expectedDim1, fDims[3], sDims[3]);
    ASSERT_EQ(nElems, cDims.elements());

    ASSERT_SUCCESS(af_create_array(&convolved, &(currGoldBar.front()),
                                   cDims.ndims(), cDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    af_array incoming_gradient = 0;
    ASSERT_SUCCESS(af_constant(&incoming_gradient, 1, cDims.ndims(),
                               cDims.get(),
                               (af_dtype)dtype_traits<T>::af_type));

    af_array filter_gradient = 0;
    ASSERT_SUCCESS(af_convolve2_gradient_nn(
        &filter_gradient, incoming_gradient, signal, filter, convolved,
        stride.ndims(), stride.get(), padding.ndims(), padding.get(),
        dilation.ndims(), dilation.get(), AF_CONV_GRADIENT_FILTER));

    af_array data_gradient = 0;
    ASSERT_SUCCESS(af_convolve2_gradient_nn(
        &data_gradient, incoming_gradient, signal, filter, convolved,
        stride.ndims(), stride.get(), padding.ndims(), padding.get(),
        dilation.ndims(), dilation.get(), AF_CONV_GRADIENT_DATA));

    vector<T> &dataGradientGold = tests[1];
    ASSERT_VEC_ARRAY_NEAR(dataGradientGold, sDims, data_gradient,
                          tolerance<T>());

    vector<T> &filterGradientGold = tests[2];
    ASSERT_VEC_ARRAY_NEAR(filterGradientGold, fDims, filter_gradient,
                          tolerance<T>());

    ASSERT_SUCCESS(af_release_array(incoming_gradient));
    ASSERT_SUCCESS(af_release_array(convolved));
    ASSERT_SUCCESS(af_release_array(signal));
    ASSERT_SUCCESS(af_release_array(filter));
    ASSERT_SUCCESS(af_release_array(filter_gradient));
    ASSERT_SUCCESS(af_release_array(data_gradient));
}

template<typename T>
class ConvolveStrided : public ::testing::Test {
   public:
    virtual void SetUp() {}
};
// create a list of types to be tested
typedef ::testing::Types<float, double, half_float::half>
    TestTypesStrided;  // TODO: integral types??

// register the type list
TYPED_TEST_SUITE(ConvolveStrided, TestTypesStrided);

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt33_s11_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig810_filt33_s11_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig81011_filt3311_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt33_s11_p11_d11) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt33_s33_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s33_p11_d11.test"),
        dim4(3, 3), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt33_s33_p11_d11) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s33_p11_d11.test"),
        dim4(3, 3), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt55_s55_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt5511_s55_p11_d11.test"),
        dim4(5, 5), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt55_s55_p11_d11) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt5511_s55_p11_d11.test"),
        dim4(5, 5), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt77_s77_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt7711_s77_p11_d11.test"),
        dim4(7, 7), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt77_s77_p11_d11) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt7711_s77_p11_d11.test"),
        dim4(7, 7), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt33_s11_p11_d22) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s11_p11_d22.test"),
        dim4(1, 1), dim4(1, 1), dim4(2, 2));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt33_s11_p11_d22) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s11_p11_d22.test"),
        dim4(1, 1), dim4(1, 1), dim4(2, 2));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt33_s11_p11_d33) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s11_p11_d33.test"),
        dim4(1, 1), dim4(1, 1), dim4(3, 3));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt33_s11_p11_d33) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3311_s11_p11_d33.test"),
        dim4(1, 1), dim4(1, 1), dim4(3, 3));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt35_s11_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3511_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt35_s11_p11_d11) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3511_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt53_s11_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt5311_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt53_s11_p11_d11) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt5311_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig1010_filt35_s31_p11_d21) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3511_s31_p11_d21.test"),
        dim4(3, 1), dim4(1, 1), dim4(2, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig1010_filt35_s31_p11_d21) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig101011_filt3511_s31_p11_d21.test"),
        dim4(3, 1), dim4(1, 1), dim4(2, 1));
}

TYPED_TEST(ConvolveStrided, Strided_sig81032_filt3334_s11_p11_d11) {
    convolve2stridedTest<TypeParam>(
        string(TEST_DIR "/convolve/sig81032_filt3334_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TYPED_TEST(ConvolveStrided, Gradient_sig81032_filt3334_s11_p11_d11) {
    convolve2GradientTest<TypeParam>(
        string(TEST_DIR "/convolve/sig81032_filt3334_s11_p11_d11.test"),
        dim4(1, 1), dim4(1, 1), dim4(1, 1));
}

TEST(ConvolveNN, ZeroPadding_Issue2817) {
    array signal = constant(1.f, 5, 5);
    array filter = constant(1 / 9.f, 3, 3);
    dim4 strides(1, 1), dilation(1, 1);
    dim4 padding(0, 0, 1, 1);

    array convolved = convolve2NN(signal, filter, strides, padding, dilation);
    ASSERT_EQ(sum<float>(abs(signal(seq(1, 3), seq(1, 3)) - convolved)) < 1E-5,
              true);
}
