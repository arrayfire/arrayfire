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
class HammingMatcher8  : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

template<typename T>
class HammingMatcher32 : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create lists of types to be tested
typedef ::testing::Types<uchar> TestTypes8;
typedef ::testing::Types<uint> TestTypes32;

// register the type list
TYPED_TEST_CASE(HammingMatcher8,  TestTypes8);
TYPED_TEST_CASE(HammingMatcher32, TestTypes32);

template<typename T>
void hammingMatcherTest(string pTestFile, int feat_dim)
{
    using af::dim4;

    vector<dim4>         numDims;
    vector<vector<uint> >   in32;
    vector<vector<uint> >  tests;

    readTests<uint, uint, uint>(pTestFile, numDims, in32, tests);

    vector<vector<T> > in(in32.size());
    for (size_t i = 0; i < in32[0].size(); i++)
        in[0].push_back((T)in32[0][i]);
    for (size_t i = 0; i < in32[1].size(); i++)
        in[1].push_back((T)in32[1][i]);

    dim4 qDims     = numDims[0];
    dim4 tDims     = numDims[1];
    af_array query = 0;
    af_array train = 0;
    af_array idx   = 0;
    af_array dist  = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&query, &(in[0].front()),
                qDims.ndims(), qDims.get(), (af_dtype)af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&train, &(in[1].front()),
                tDims.ndims(), tDims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_hamming_matcher(&idx, &dist, query, train, feat_dim, 1));

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outIdx,  idx));
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outDist, dist));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outIdx;
    delete[] outDist;
    ASSERT_EQ(AF_SUCCESS, af_release_array(query));
    ASSERT_EQ(AF_SUCCESS, af_release_array(train));
    ASSERT_EQ(AF_SUCCESS, af_release_array(idx));
    ASSERT_EQ(AF_SUCCESS, af_release_array(dist));
}

TYPED_TEST(HammingMatcher8, Hamming_500_5000_Dim0)
{
    hammingMatcherTest<TypeParam>(string(TEST_DIR"/hamming/hamming_500_5000_dim0_u8.test"), 0);
}

TYPED_TEST(HammingMatcher8, Hamming_500_5000_Dim1)
{
    hammingMatcherTest<TypeParam>(string(TEST_DIR"/hamming/hamming_500_5000_dim1_u8.test"), 1);
}

TYPED_TEST(HammingMatcher32, Hamming_500_5000_Dim0)
{
    hammingMatcherTest<TypeParam>(string(TEST_DIR"/hamming/hamming_500_5000_dim0_u32.test"), 0);
}

TYPED_TEST(HammingMatcher32, Hamming_500_5000_Dim1)
{
    hammingMatcherTest<TypeParam>(string(TEST_DIR"/hamming/hamming_500_5000_dim1_u32.test"), 1);
}

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(HammingMatcher, CPP)
{
    using af::array;
    using af::dim4;

    vector<dim4>         numDims;
    vector<vector<uint> >     in;
    vector<vector<uint> >  tests;

    readTests<uint, uint, uint>(TEST_DIR"/hamming/hamming_500_5000_dim0_u32.test", numDims, in, tests);

    dim4 qDims     = numDims[0];
    dim4 tDims     = numDims[1];

    array query(qDims, &(in[0].front()));
    array train(tDims, &(in[1].front()));

    array idx, dist;
    hammingMatcher(idx, dist, query, train, 0, 1);

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    idx.host(outIdx);
    dist.host(outDist);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outIdx;
    delete[] outDist;
}
