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
#include <af/defines.h>
#include <af/traits.hpp>
#include <af/data.h>

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::generate;
using std::cout;
using std::endl;
using std::ostream_iterator;
using af::dtype_traits;

void testGeneralIndexOneArray(string pTestFile, const dim_type ndims, af_index_t* indexers, int arrayDim)
{
    vector<af::dim4>        numDims;
    vector< vector<float> >      in;
    vector< vector<float> >   tests;

    readTestsFromFile<float, float>(pTestFile, numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af_array outArray  = 0;
    af_array inArray   = 0;
    af_array idxArray  = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexers[arrayDim].mIndexer.arr = idxArray;

    ASSERT_EQ(AF_SUCCESS, af_index_gen(&outArray, inArray, ndims, indexers));

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(idxArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

TEST(GeneralIndex, SSSA)
{
    af_index_t indexers[4];
    indexers[0].mIndexer.seq = af_make_seq(0, 3, 1);
    indexers[1].mIndexer.seq = af_make_seq(0, 1, 1);
    indexers[2].mIndexer.seq = af_make_seq(1, 2, 1);
    indexers[0].mIsSeq = true;
    indexers[1].mIsSeq = true;
    indexers[2].mIsSeq = true;
    indexers[3].mIsSeq = false;

    testGeneralIndexOneArray(string(TEST_DIR"/gen_index/s0_3s0_1s1_2a.test"), 4, indexers, 3);
}

TEST(GeneralIndex, ASSS)
{
    af_index_t indexers[4];
    indexers[1].mIndexer.seq = af_make_seq(0, 9, 1);
    indexers[2].mIndexer.seq = af_span;
    indexers[3].mIndexer.seq = af_span;
    indexers[0].mIsSeq = false;
    indexers[1].mIsSeq = true;
    indexers[2].mIsSeq = true;
    indexers[3].mIsSeq = true;

    testGeneralIndexOneArray(string(TEST_DIR"/gen_index/as0_9s0_ns0_n.test"), 4, indexers, 0);
}

TEST(GeneralIndex, SASS)
{
    af_index_t indexers[2];
    indexers[0].mIndexer.seq = af_make_seq(10, 40, 1);
    indexers[0].mIsSeq = true;
    indexers[1].mIsSeq = false;

    testGeneralIndexOneArray(string(TEST_DIR"/gen_index/s0_9as0_ns0_n.test"), 2, indexers, 1);
}

TEST(GeneralIndex, AASS)
{
    vector<af::dim4>        numDims;
    vector< vector<float> >      in;
    vector< vector<float> >   tests;

    readTestsFromFile<float, float>(string(TEST_DIR"/gen_index/aas0_ns0_n.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af::dim4 dims2     = numDims[2];
    af_array outArray  = 0;
    af_array inArray   = 0;
    af_array idxArray0 = 0;
    af_array idxArray1 = 0;

    af_index_t indexers[2];

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray0, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexers[0].mIsSeq = false;
    indexers[0].mIndexer.arr = idxArray0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray1, &(in[2].front()),
                dims2.ndims(), dims2.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexers[1].mIsSeq = false;
    indexers[1].mIndexer.arr = idxArray1;

    ASSERT_EQ(AF_SUCCESS, af_index_gen(&outArray, inArray, 2, indexers));

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(idxArray0));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(idxArray1));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}
