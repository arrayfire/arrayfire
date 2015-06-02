/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <af/signal.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

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
    APPROX2_ARGS(Approx2ArgsInterpBilinear,    approx2,         0, AF_INTERP_BILINEAR, AF_ERR_ARG);
    APPROX2_ARGS(Approx2ArgsInterpCubic,       approx2,         0, AF_INTERP_CUBIC,    AF_ERR_ARG);

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
    if (noDoubleTests<float>()) return;
    const unsigned resultIdx = 1;
#define BT af::dtype_traits<float>::base_type
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx2.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];
    af::dim4 qdims = numDims[2];

    af::array input(idims,&(in[0].front()));
    af::array pos0(pdims,&(in[1].front()));
    af::array pos1(qdims,&(in[2].front()));
    af::array output = af::approx2(input, pos0, pos1, AF_INTERP_LINEAR, 0);

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
