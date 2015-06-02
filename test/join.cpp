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
#include <af/index.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

template<typename T>
class Join : public ::testing::Test
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
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int, intl, uintl, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Join, TestTypes);

template<typename T>
void joinTest(string pTestFile, const unsigned dim, const unsigned in0, const unsigned in1, const unsigned resultIdx,
        bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, int>(pTestFile,numDims,in,tests);

    af::dim4 i0dims = numDims[in0];
    af::dim4 i1dims = numDims[in1];

    af_array in0Array = 0;
    af_array in1Array = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[in0].front()), i0dims.ndims(), i0dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&in0Array, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&in0Array, &(in[in0].front()), i0dims.ndims(), i0dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[in1].front()), i1dims.ndims(), i1dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&in1Array, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&in1Array, &(in[in1].front()), i1dims.ndims(), i1dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_join(&outArray, dim, in0Array, in1Array));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

    if(in0Array  != 0) af_release_array(in0Array);
    if(in1Array  != 0) af_release_array(in1Array);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define JOIN_INIT(desc, file, dim, in0, in1, resultIdx)                                     \
    TYPED_TEST(Join, desc)                                                                  \
    {                                                                                       \
        joinTest<TypeParam>(string(TEST_DIR"/join/"#file".test"), dim, in0, in1, resultIdx);\
    }

    JOIN_INIT(JoinBig0, join_big, 0, 0, 1, 0);
    JOIN_INIT(JoinBig1, join_big, 1, 0, 2, 1);
    JOIN_INIT(JoinBig2, join_big, 2, 0, 3, 2);

    JOIN_INIT(JoinSmall0, join_small, 0, 0, 1, 0);
    JOIN_INIT(JoinSmall1, join_small, 1, 0, 2, 1);
    JOIN_INIT(JoinSmall2, join_small, 2, 0, 3, 2);

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Join, CPP)
{
    if (noDoubleTests<float>()) return;

    const unsigned resultIdx = 2;
    const unsigned dim = 2;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, int>(string(TEST_DIR"/join/join_big.test"),numDims,in,tests);

    af::dim4 i0dims = numDims[0];
    af::dim4 i1dims = numDims[3];

    af::array input0(i0dims, &(in[0].front()));
    af::array input1(i1dims, &(in[3].front()));

    af::array output = af::join(dim, input0, input1);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;
}

TEST(JoinMany0, CPP)
{
    if (noDoubleTests<float>()) return;

    af::array a0 = af::randu(10, 5);
    af::array a1 = af::randu(20, 5);
    af::array a2 = af::randu(5, 5);

    af::array output = af::join(0, a0, a1, a2);
    af::array gold = af::join(0, a0, af::join(0, a1, a2));


    ASSERT_EQ(af::sum<float>(output - gold), 0);
}

TEST(JoinMany1, CPP)
{
    if (noDoubleTests<float>()) return;

    af::array a0 = af::randu(20, 200);
    af::array a1 = af::randu(20, 400);
    af::array a2 = af::randu(20, 10);
    af::array a3 = af::randu(20, 100);

    int dim = 1;
    af::array output = af::join(dim, a0, a1, a2, a3);
    af::array gold = af::join(dim, a0, af::join(dim, a1, af::join(dim, a2, a3)));

    ASSERT_EQ(af::sum<float>(output - gold), 0);
}
