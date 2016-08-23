/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <af/array.h>
#include <af/complex.h>
#include <af/signal.h>
#include <af/index.h>
#include <af/traits.hpp>
#include <af/dim4.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::abs;
using af::cfloat;
using af::cdouble;

template<typename T>
class Approx1 : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back(af_make_seq(0, 4, 1));
            subMat0.push_back(af_make_seq(2, 6, 1));
            subMat0.push_back(af_make_seq(0, 2, 1));
        }
        vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// register the type list
TYPED_TEST_CASE(Approx1, TestTypes);

template<typename T>
void approx1Test(string pTestFile, const unsigned resultIdx, const af_interp_type method, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    typedef typename af::dtype_traits<T>::base_type BT;
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];

    af_array inArray = 0;
    af_array posArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define APPROX1_INIT(desc, file, resultIdx, method)                                         \
    TYPED_TEST(Approx1, desc)                                                               \
    {                                                                                       \
        approx1Test<TypeParam>(string(TEST_DIR"/approx/"#file".test"), resultIdx, method);  \
    }

    APPROX1_INIT(Approx1Nearest, approx1, 0, AF_INTERP_NEAREST);
    APPROX1_INIT(Approx1Linear,  approx1, 1, AF_INTERP_LINEAR);

template<typename T>
void approx1CubicTest(string pTestFile, const unsigned resultIdx, const af_interp_type method, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    typedef typename af::dtype_traits<T>::base_type BT;
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];

    af_array inArray = 0;
    af_array posArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;

    float max = real(outData[0]), min = real(outData[0]);
    for(int i=1; i < nElems; ++i) {
        min = (real(outData[i]) < min) ? real(outData[i]) : min;
        max = (real(outData[i]) > max) ? real(outData[i]) : max;
    }
    float range = max - min;
    ASSERT_GT(range, 0.f);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        double integral;
        //test that control points are exact
        if((std::modf(in[1][elIter], &integral) < 0.001) || (std::modf(in[1][elIter], &integral) > 0.999))  {
            ret = abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
        } else {
            //match intermediate values withing a threshold
            ret = abs(tests[resultIdx][elIter] - outData[elIter]) < 0.035 * range;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
        }
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define APPROX1_INIT_CUBIC(desc, file, resultIdx, method)                                       \
    TYPED_TEST(Approx1, desc)                                                                   \
    {                                                                                           \
        approx1CubicTest<TypeParam>(string(TEST_DIR"/approx/"#file".test"), resultIdx, method); \
    }

APPROX1_INIT_CUBIC(Approx1Cubic, approx1_cubic, 0, AF_INTERP_CUBIC);

///////////////////////////////////////////////////////////////////////////////
// Test Argument Failure Cases
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void approx1ArgsTest(string pTestFile, const unsigned resultIdx, const af_interp_type method, const af_err err)
{
    if (noDoubleTests<T>()) return;
    typedef typename af::dtype_traits<T>::base_type BT;
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];

    af_array inArray  = 0;
    af_array posArray = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));

    ASSERT_EQ(err, af_approx1(&outArray, inArray, posArray, method, 0));

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
}

#define APPROX1_ARGS(desc, file, resultIdx, method, err)                                            \
    TYPED_TEST(Approx1, desc)                                                                       \
    {                                                                                               \
        approx1ArgsTest<TypeParam>(string(TEST_DIR"/approx/"#file".test"), resultIdx, method, err); \
    }

    APPROX1_ARGS(Approx1NearestArgsPos2D, approx1_pos2d, 0, AF_INTERP_NEAREST, AF_ERR_SIZE);
    APPROX1_ARGS(Approx1LinearArgsPos2D, approx1_pos2d, 1, AF_INTERP_LINEAR, AF_ERR_SIZE);
    APPROX1_ARGS(Approx1ArgsInterpBilinear, approx1, 0, AF_INTERP_BILINEAR, AF_ERR_ARG);

template<typename T>
void approx1ArgsTestPrecision(string pTestFile, const unsigned resultIdx, const af_interp_type method)
{
    if (noDoubleTests<T>()) return;
    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];

    af_array inArray  = 0;
    af_array posArray = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    if((af_dtype) af::dtype_traits<T>::af_type == c32 ||
       (af_dtype) af::dtype_traits<T>::af_type == c64) {
        ASSERT_EQ(AF_ERR_ARG, af_approx1(&outArray, inArray, posArray, method, 0));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_approx1(&outArray, inArray, posArray, method, 0));
    }

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
}

#define APPROX1_ARGSP(desc, file, resultIdx, method)                                                    \
    TYPED_TEST(Approx1, desc)                                                                           \
    {                                                                                                   \
        approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR"/approx/"#file".test"), resultIdx, method); \
    }

    APPROX1_ARGSP(Approx1NearestArgsPrecision, approx1, 0, AF_INTERP_NEAREST);
    APPROX1_ARGSP(Approx1LinearArgsPrecision, approx1, 1, AF_INTERP_LINEAR);
    APPROX1_ARGSP(Approx1CubicArgsPrecision, approx1_cubic, 2, AF_INTERP_CUBIC);

//////////////////////////////////////// CPP //////////////////////////////////
//
TEST(Approx1, CPP)
{
    if (noDoubleTests<float>()) return;
    const unsigned resultIdx = 1;
    const af_interp_type method = AF_INTERP_LINEAR;
#define BT af::dtype_traits<float>::base_type
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx1.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];

    af::array input(idims, &(in[0].front()));
    af::array pos(pdims, &(in[1].front()));

    af::array output = approx1(input, pos, method, 0);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx1, CPPNearestBatch)
{
    if (noDoubleTests<float>()) return;

    af::array input = af::randu(600, 10);
    af::array pos   = input.dims(0) * af::randu(100, 10);

    af::array outBatch = af::approx1(input, pos, AF_INTERP_NEAREST);

    af::array outSerial(pos.dims());
    for(int i = 0; i < pos.dims(1); i++) {
        outSerial(af::span, i) = af::approx1(input(af::span, i), pos(af::span, i), AF_INTERP_NEAREST);
    }

    af::array outGFOR(pos.dims());
    gfor(af::seq i, pos.dims(1)) {
        outGFOR(af::span, i) = af::approx1(input(af::span, i), pos(af::span, i), AF_INTERP_NEAREST);
    }

    ASSERT_NEAR(0, af::sum<double>(af::abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, af::sum<double>(af::abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPLinearBatch)
{
    if (noDoubleTests<float>()) return;

    af::array input = af::iota(af::dim4(10000, 20), c32);
    af::array pos   = input.dims(0) * af::randu(50000, 20);

    af::array outBatch = af::approx1(input, pos, AF_INTERP_LINEAR);

    af::array outSerial(pos.dims());
    for(int i = 0; i < pos.dims(1); i++) {
        outSerial(af::span, i) = af::approx1(input(af::span, i), pos(af::span, i), AF_INTERP_LINEAR);
    }

    af::array outGFOR(pos.dims());
    gfor(af::seq i, pos.dims(1)) {
        outGFOR(af::span, i) = af::approx1(input(af::span, i), pos(af::span, i), AF_INTERP_LINEAR);
    }

    ASSERT_NEAR(0, af::sum<double>(af::abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, af::sum<double>(af::abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPCubicBatch)
{
    if (noDoubleTests<float>()) return;

    af::array input = af::iota(af::dim4(10000, 20), c32);
    af::array pos   = input.dims(0) * af::randu(50000, 20);

    af::array outBatch = af::approx1(input, pos, AF_INTERP_CUBIC);

    af::array outSerial(pos.dims());
    for(int i = 0; i < pos.dims(1); i++) {
        outSerial(af::span, i) = af::approx1(input(af::span, i), pos(af::span, i), AF_INTERP_CUBIC);
    }

    af::array outGFOR(pos.dims());
    gfor(af::seq i, pos.dims(1)) {
        outGFOR(af::span, i) = af::approx1(input(af::span, i), pos(af::span, i), AF_INTERP_CUBIC);
    }

    ASSERT_NEAR(0, af::sum<double>(af::abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, af::sum<double>(af::abs(outBatch - outGFOR)), 1e-3);
}
