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

void testGeneralIndexOneArray(string pTestFile, const dim_t ndims, af_index_t* indexs, int arrayDim)
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
    indexs[arrayDim].idx.arr = idxArray;

    ASSERT_EQ(AF_SUCCESS, af_index_gen(&outArray, inArray, ndims, indexs));

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(idxArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TEST(GeneralIndex, SSSA)
{
    af_index_t indexs[4];
    indexs[0].idx.seq = af_make_seq(0, 3, 1);
    indexs[1].idx.seq = af_make_seq(0, 1, 1);
    indexs[2].idx.seq = af_make_seq(1, 2, 1);
    indexs[0].isSeq = true;
    indexs[1].isSeq = true;
    indexs[2].isSeq = true;
    indexs[3].isSeq = false;

    testGeneralIndexOneArray(string(TEST_DIR"/gen_index/s0_3s0_1s1_2a.test"), 4, indexs, 3);
}

TEST(GeneralIndex, ASSS)
{
    af_index_t indexs[4];
    indexs[1].idx.seq = af_make_seq(0, 9, 1);
    indexs[2].idx.seq = af_span;
    indexs[3].idx.seq = af_span;
    indexs[0].isSeq = false;
    indexs[1].isSeq = true;
    indexs[2].isSeq = true;
    indexs[3].isSeq = true;

    testGeneralIndexOneArray(string(TEST_DIR"/gen_index/as0_9s0_ns0_n.test"), 4, indexs, 0);
}

TEST(GeneralIndex, SASS)
{
    af_index_t indexs[2];
    indexs[0].idx.seq = af_make_seq(10, 40, 1);
    indexs[0].isSeq = true;
    indexs[1].isSeq = false;

    testGeneralIndexOneArray(string(TEST_DIR"/gen_index/s0_9as0_ns0_n.test"), 2, indexs, 1);
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

    af_index_t indexs[2];

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray0, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexs[0].isSeq = false;
    indexs[0].idx.arr = idxArray0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray1, &(in[2].front()),
                dims2.ndims(), dims2.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexs[1].isSeq = false;
    indexs[1].idx.arr = idxArray1;

    ASSERT_EQ(AF_SUCCESS, af_index_gen(&outArray, inArray, 2, indexs));

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(idxArray0));
    ASSERT_EQ(AF_SUCCESS, af_release_array(idxArray1));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TEST(GeneralIndex, CPP_ASNN)
{
    using namespace af;
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a = randu(nx, ny);
    array idx = where(randu(nx) > 0.5);
    array b = a(idx, seq(st, en));

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);

    float *hA = a.host<float>();
    uint  *hIdx = idx.host<uint>();
    float *hB = b.host<float>();


    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + (st + j) * nx;
        float *hBt = hB + j * nxb;
        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[hIdx[i]], hBt[i])
                << "at " << i << " " << j << std::endl;
        }
    }

    delete[] hA;
    delete[] hB;
    delete[] hIdx;
}

TEST(GeneralIndex, CPP_SANN)
{
    using namespace af;
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a = randu(nx, ny);
    array idx = where(randu(ny) > 0.5);
    array b = a(seq(st, en), idx);

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);

    float *hA = a.host<float>();
    uint  *hIdx = idx.host<uint>();
    float *hB = b.host<float>();

    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + hIdx[j] * nx;
        float *hBt = hB + j * nxb;

        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[i + st], hBt[i])
            << "at " << i << " " << j << std::endl;
        }
    }

    delete[] hA;
    delete[] hB;
    delete[] hIdx;
}

TEST(GeneralIndex, CPP_SSAN)
{
    using namespace af;
    const int nx = 100;
    const int ny = 100;
    const int nz = 100;
    const int st = 20;
    const int en = 85;

    array a = randu(nx, ny, nz);
    array idx = where(randu(nz) > 0.5);
    array b = a(seq(st, en), span, idx);

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);
    const int nzb = b.dims(2);

    float *hA = a.host<float>();
    uint  *hIdx = idx.host<uint>();
    float *hB = b.host<float>();

    for (int k = 0; k < nzb; k++) {
        float *hAt = hA + hIdx[k] * nx * ny;
        float *hBt = hB + k * nxb * nyb;

        for (int j = 0; j < nyb; j++) {
            for (int i = 0; i < nxb; i++) {
                ASSERT_EQ(hAt[j * nx  + i + st], hBt[j * nxb + i])
                    << "at " << i << " " << j << " " << k << std::endl;
            }
        }
    }

    delete[] hA;
    delete[] hB;
    delete[] hIdx;
}

TEST(GeneralIndex, CPP_AANN)
{
    using namespace af;
    const int nx = 1000;
    const int ny = 1000;

    array a = randu(nx, ny);
    array idx0 = where(randu(nx) > 0.5);
    array idx1 = where(randu(ny) > 0.5);
    array b = a(idx0, idx1);

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);

    float *hA = a.host<float>();
    uint  *hIdx0 = idx0.host<uint>();
    uint  *hIdx1 = idx1.host<uint>();
    float *hB = b.host<float>();


    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + hIdx1[j] * nx;
        float *hBt = hB + j * nxb;
        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[hIdx0[i]], hBt[i])
                << "at " << i << " " << j << std::endl;
        }
    }

    delete[] hA;
    delete[] hB;
    delete[] hIdx0;
    delete[] hIdx1;
}
