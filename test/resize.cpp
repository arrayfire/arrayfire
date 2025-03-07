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
#include <iostream>
#include <string>
#include <vector>

using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Resize : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

template<typename T>
class ResizeI : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));

        subMat1.push_back(af_make_seq(0, 5, 1));
        subMat1.push_back(af_make_seq(0, 5, 1));
        subMat1.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
    vector<af_seq> subMat1;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypesF;
typedef ::testing::Types<int, unsigned, intl, uintl, unsigned char, char, short,
                         ushort>
    TestTypesI;

// register the type list
TYPED_TEST_SUITE(Resize, TestTypesF);
TYPED_TEST_SUITE(ResizeI, TestTypesI);

TYPED_TEST(Resize, InvalidDims) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    vector<TypeParam> in(8 * 8);

    af_array inArray  = 0;
    af_array outArray = 0;

    dim4 dims = dim4(8, 8, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<TypeParam>::af_type));
    ASSERT_EQ(AF_ERR_SIZE,
              af_resize(&outArray, inArray, 0, 0, AF_INTERP_NEAREST));
    ASSERT_SUCCESS(af_release_array(inArray));
}

template<typename T>
void compare(T test, T out, double err, size_t i) {
    ASSERT_EQ(abs(test - out) < err, true) << "at: " << i << endl
                                           << "for test = : " << test << endl
                                           << "out data = : " << out << endl;
}

template<>
void compare<uintl>(uintl test, uintl out, double err, size_t i) {
    ASSERT_EQ(((intl)test - (intl)out) < err, true)
        << "at: " << i << endl
        << "for test = : " << test << endl
        << "out data = : " << out << endl;
}

template<>
void compare<uint>(uint test, uint out, double err, size_t i) {
    ASSERT_EQ(((int)test - (int)out) < err, true)
        << "at: " << i << endl
        << "for test = : " << test << endl
        << "out data = : " << out << endl;
}

template<>
void compare<uchar>(uchar test, uchar out, double err, size_t i) {
    ASSERT_EQ(((int)test - (int)out) < err, true)
        << "at: " << i << endl
        << "for test = : " << test << endl
        << "out data = : " << out << endl;
}

template<typename T>
void resizeTest(string pTestFile, const unsigned resultIdx, const dim_t odim0,
                const dim_t odim1, const af_interp_type method,
                bool isSubRef = false, const vector<af_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 dims = numDims[0];

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;
    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(in[0].front()),
                                       dims.ndims(), dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                       dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_resize(&outArray, inArray, odim0, odim1, method));

    // Get result
    dim4 odims(odim0, odim1, dims[2], dims[3]);
    T* outData = new T[odims.elements()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        compare<T>(tests[resultIdx][elIter], outData[elIter], 0.0001, elIter);
    }

    // Delete
    delete[] outData;

    if (inArray != 0) af_release_array(inArray);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

///////////////////////////////////////////////////////////////////////////////
// Float Types
///////////////////////////////////////////////////////////////////////////////
TYPED_TEST(Resize, Resize3CSquareUpNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 0, 16, 16,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize3CSquareUpLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 1, 16, 16,
                          AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize3CSquareDownNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 2, 4, 4,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize3CSquareDownLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 3, 4, 4,
                          AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize3CSquareUpNearestSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 4, 10, 10,
                          AF_INTERP_NEAREST, true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize3CSquareUpLinearSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 5, 10, 10,
                          AF_INTERP_BILINEAR, true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize3CSquareDownNearestSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 6, 3, 3,
                          AF_INTERP_NEAREST, true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize3CSquareDownLinearSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 7, 3, 3,
                          AF_INTERP_BILINEAR, true, &(this->subMat0));
}

TYPED_TEST(Resize, Resize1CRectangleUpNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/rectangle.test"), 0, 12, 16,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CRectangleUpLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/rectangle.test"), 1, 12, 16,
                          AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize1CRectangleDownNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/rectangle.test"), 2, 6, 2,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CRectangleDownLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/rectangle.test"), 3, 6, 2,
                          AF_INTERP_BILINEAR);
}

///////////////////////////////////////////////////////////////////////////////
// Interger Types
///////////////////////////////////////////////////////////////////////////////
TYPED_TEST(ResizeI, Resize3CSquareUpNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 0, 16, 16,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize3CSquareUpLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 1, 16, 16,
                          AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize3CSquareDownNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 2, 4, 4,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize3CSquareDownLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 3, 4, 4,
                          AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize3CSquareUpNearestSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 4, 10, 10,
                          AF_INTERP_NEAREST, true, &(this->subMat0));
}

TYPED_TEST(ResizeI, Resize3CSquareUpLinearSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 5, 10, 10,
                          AF_INTERP_BILINEAR, true, &(this->subMat0));
}

TYPED_TEST(ResizeI, Resize3CSquareDownNearestSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 6, 3, 3,
                          AF_INTERP_NEAREST, true, &(this->subMat0));
}

TYPED_TEST(ResizeI, Resize3CSquareDownLinearSubref) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/square.test"), 8, 3, 3,
                          AF_INTERP_BILINEAR, true, &(this->subMat1));
}

///////////////////////////////////////////////////////////////////////////////
// Float Types
///////////////////////////////////////////////////////////////////////////////
TYPED_TEST(Resize, Resize1CLargeUpNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 0, 256, 256,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CLargeUpLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 1, 256, 256,
                          AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, Resize1CLargeDownNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 2, 32, 32,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(Resize, Resize1CLargeDownLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 3, 32, 32,
                          AF_INTERP_BILINEAR);
}

///////////////////////////////////////////////////////////////////////////////
// Integer Types
///////////////////////////////////////////////////////////////////////////////
TYPED_TEST(ResizeI, Resize1CLargeUpNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 0, 256, 256,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize1CLargeUpLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 1, 256, 256,
                          AF_INTERP_BILINEAR);
}

TYPED_TEST(ResizeI, Resize1CLargeDownNearest) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 2, 32, 32,
                          AF_INTERP_NEAREST);
}

TYPED_TEST(ResizeI, Resize1CLargeDownLinear) {
    resizeTest<TypeParam>(string(TEST_DIR "/resize/large.test"), 3, 32, 32,
                          AF_INTERP_BILINEAR);
}

template<typename T>
void resizeArgsTest(af_err err, string pTestFile, const dim4 odims,
                    const af_interp_type method) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 dims = numDims[0];

    af_array inArray  = 0;
    af_array outArray = 0;
    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_EQ(err, af_resize(&outArray, inArray, odims[0], odims[1], method));

    if (inArray != 0) af_release_array(inArray);
    if (outArray != 0) af_release_array(outArray);
}

TYPED_TEST(Resize, InvalidArgsDims0) {
    dim4 dims(0, 5, 2, 1);
    resizeArgsTest<TypeParam>(AF_ERR_SIZE,
                              string(TEST_DIR "/resize/square.test"), dims,
                              AF_INTERP_BILINEAR);
}

TYPED_TEST(Resize, InvalidArgsMethod) {
    dim4 dims(10, 10, 1, 1);
    resizeArgsTest<TypeParam>(AF_ERR_ARG,
                              string(TEST_DIR "/resize/square.test"), dims,
                              AF_INTERP_CUBIC);
}

///////////////////////////////// CPP ////////////////////////////////////
//

using af::array;
using af::constant;
using af::max;
using af::seq;
using af::span;

TEST(Resize, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/resize/square.test"),
                                   numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output = resize(input, 16, 16);

    dim4 goldDims(16, 16, dims[2], dims[3]);
    ASSERT_VEC_ARRAY_NEAR(tests[0], goldDims, output, 0.0001);
}

TEST(ResizeScale1, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/resize/square.test"),
                                   numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output = resize(2.f, input);

    dim4 goldDims(16, 16, dims[2], dims[3]);
    ASSERT_VEC_ARRAY_NEAR(tests[0], goldDims, output, 0.0001);
}

TEST(ResizeScale2, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/resize/square.test"),
                                   numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output = resize(2.f, 2.f, input);

    dim4 goldDims(16, 16, dims[2], dims[3]);
    ASSERT_VEC_ARRAY_NEAR(tests[0], goldDims, output, 0.0001);
}

TEST(Resize, ExtractGFOR) {
    dim4 dims = dim4(100, 100, 3);
    array A   = round(100 * randu(dims));
    array B   = constant(0, 200, 200, 3);

    gfor(seq ii, 3) { B(span, span, ii) = resize(A(span, span, ii), 200, 200); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = resize(A(span, span, ii), 200, 200);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
