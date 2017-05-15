/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/signal.h>
#include <af/traits.hpp>

#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <iostream>
#include <string>
#include <vector>

using af::abs;
using af::approx2;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::randu;
using af::seq;
using af::span;
using af::sum;

using std::abs;
using std::string;
using std::vector;

template<typename T>
class Approx2 : public ::testing::Test
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
TYPED_TEST_CASE(Approx2, TestTypes);

template<typename T>
void approx2Test(string pTestFile, const unsigned resultIdx, const af_interp_type method, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;
    typedef typename af::dtype_traits<T>::base_type BT;
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];
    af::dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(pos0Array != 0) af_release_array(pos0Array);
    if(pos1Array != 0) af_release_array(pos1Array);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define APPROX2_INIT(desc, file, resultIdx, method)                               \
    TYPED_TEST(Approx2, desc)                                                                    \
    {                                                                                           \
        approx2Test<TypeParam>(string(TEST_DIR"/approx/"#file".test"), resultIdx, method);\
    }

    APPROX2_INIT(Approx2Nearest, approx2, 0, AF_INTERP_NEAREST);
    APPROX2_INIT(Approx2Linear, approx2, 1, AF_INTERP_LINEAR);
    APPROX2_INIT(Approx2NearestBatch, approx2_batch, 0, AF_INTERP_NEAREST);
    APPROX2_INIT(Approx2LinearBatch, approx2_batch, 1, AF_INTERP_LINEAR);

///////////////////////////////////////////////////////////////////////////////
// Test Argument Failure Cases
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void approx2ArgsTest(string pTestFile, const unsigned resultIdx, const af_interp_type method, const af_err err)
{
    if (noDoubleTests<T>()) return;
    typedef typename af::dtype_traits<T>::base_type BT;
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];
    af::dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));

    ASSERT_EQ(err, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));

    if(inArray   != 0) af_release_array(inArray);
    if(pos0Array != 0) af_release_array(pos0Array);
    if(pos1Array != 0) af_release_array(pos1Array);
    if(outArray  != 0) af_release_array(outArray);
}

#define APPROX2_ARGS(desc, file, resultIdx, method, err)                                            \
    TYPED_TEST(Approx2, desc)                                                                       \
    {                                                                                               \
        approx2ArgsTest<TypeParam>(string(TEST_DIR"/approx/"#file".test"), resultIdx, method, err); \
    }

    APPROX2_ARGS(Approx2NearestArgsPos3D,      approx2_pos3d,   0, AF_INTERP_NEAREST,  AF_ERR_SIZE);
    APPROX2_ARGS(Approx2LinearArgsPos3D,       approx2_pos3d,   1, AF_INTERP_LINEAR,   AF_ERR_SIZE);
    APPROX2_ARGS(Approx2NearestArgsPosUnequal, approx2_unequal, 0, AF_INTERP_NEAREST,  AF_ERR_SIZE);

template<typename T>
void approx2ArgsTestPrecision(string pTestFile, const unsigned resultIdx, const af_interp_type method)
{
    if (noDoubleTests<T>()) return;
    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];
    af::dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) af::dtype_traits<T>::af_type));


    if((af_dtype) af::dtype_traits<T>::af_type == c32 ||
       (af_dtype) af::dtype_traits<T>::af_type == c64) {
        ASSERT_EQ(AF_ERR_ARG, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));
    }

    if(inArray   != 0) af_release_array(inArray);
    if(pos0Array != 0) af_release_array(pos0Array);
    if(pos1Array != 0) af_release_array(pos1Array);
    if(outArray  != 0) af_release_array(outArray);
}

#define APPROX2_ARGSP(desc, file, resultIdx, method)                                    \
    TYPED_TEST(Approx2, desc)                                                           \
    {                                                                                   \
        approx2ArgsTestPrecision<TypeParam>(string(TEST_DIR"/approx/"#file".test"),     \
                                            resultIdx, method);                         \
    }

    APPROX2_ARGSP(Approx2NearestArgsPrecision, approx2, 0, AF_INTERP_NEAREST);
    APPROX2_ARGSP(Approx2LinearArgsPrecision, approx2, 1, AF_INTERP_LINEAR);


//////////////////////////////////// CPP ////////////////////////////////////
//
TEST(Approx2, CPP)
{
    const unsigned resultIdx = 1;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx2.test"),numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    array input(idims,&(in[0].front()));
    array pos0(pdims,&(in[1].front()));
    array pos1(qdims,&(in[2].front()));
    array output = approx2(input, pos0, pos1, AF_INTERP_LINEAR, 0);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx2Cubic, CPP)
{
    const unsigned resultIdx = 0;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx2_cubic.test"),numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    array input(idims,&(in[0].front()));
    input = input.T();
    array pos0(pdims,&(in[1].front()));
    array pos1(qdims,&(in[2].front()));
    pos0 = tile(pos0, 1, pos0.dims(0));
    pos1 = tile(pos1.T(), pos1.dims(0));
    array output = approx2(input, pos0, pos1, AF_INTERP_BICUBIC_SPLINE, 0).T();

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;

    float max = real(outData[0]), min = real(outData[0]);
    for(int i=1; i < (int)nElems; ++i) {
        min = (real(outData[i]) < min) ? real(outData[i]) : min;
        max = (real(outData[i]) > max) ? real(outData[i]) : max;
    }
    float range = max - min;
    ASSERT_GT(range, 0.f);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.01 * range);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx2, CPPNearestBatch)
{
    array input = randu(200, 100, 10);
    array pos   = input.dims(0) * randu(100, 100, 10);
    array qos   = input.dims(1) * randu(100, 100, 10);

    array outBatch = approx2(input, pos, qos, AF_INTERP_NEAREST);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(2); i++) {
        outSerial(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_NEAREST);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(2)) {
        outGFOR(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_NEAREST);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx2, CPPLinearBatch)
{
    array input = randu(200, 100, 10);
    array pos   = input.dims(0) * randu(100, 100, 10);
    array qos   = input.dims(1) * randu(100, 100, 10);

    array outBatch = approx2(input, pos, qos, AF_INTERP_LINEAR);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(2); i++) {
        outSerial(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_LINEAR);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(2)) {
        outGFOR(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_LINEAR);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}
