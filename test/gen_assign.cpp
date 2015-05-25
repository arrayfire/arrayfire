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

void testGeneralAssignOneArray(string pTestFile, const dim_t ndims, af_index_t* indexs, int arrayDim)
{
    vector<af::dim4>        numDims;
    vector< vector<float> >      in;
    vector< vector<float> >   tests;

    readTestsFromFile<float, float>(pTestFile, numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af::dim4 dims2     = numDims[2];
    af_array outArray  = 0;
    af_array rhsArray  = 0;
    af_array lhsArray  = 0;
    af_array idxArray  = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&lhsArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&rhsArray, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray, &(in[2].front()),
                dims2.ndims(), dims2.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexs[arrayDim].idx.arr = idxArray;

    ASSERT_EQ(AF_SUCCESS, af_assign_gen(&outArray, lhsArray, ndims, indexs, rhsArray));

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(rhsArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(lhsArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(idxArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TEST(GeneralAssign, ASSS)
{
    af_index_t indexs[2];
    indexs[1].idx.seq = af_make_seq(0, 9, 1);
    indexs[0].isSeq = false;
    indexs[1].isSeq = true;

    testGeneralAssignOneArray(string(TEST_DIR"/gen_assign/as0_9s0_ns0_n.test"), 2, indexs, 0);
}

TEST(GeneralAssign, SASS)
{
    af_index_t indexs[2];
    indexs[0].idx.seq = af_make_seq(10, 14, 1);
    indexs[0].isSeq = true;
    indexs[1].isSeq = false;

    testGeneralAssignOneArray(string(TEST_DIR"/gen_assign/s10_14as0_ns0_n.test"), 2, indexs, 1);
}

TEST(GeneralAssign, SSSS)
{
    vector<af::dim4>        numDims;
    vector< vector<float> >      in;
    vector< vector<float> >   tests;

    readTestsFromFile<float, float>(string(TEST_DIR"/gen_assign/s10_14s0_9s0_ns0_n.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af_array outArray  = 0;
    af_array rhsArray  = 0;
    af_array lhsArray  = 0;

    af_index_t indexs[2];
    indexs[0].idx.seq = af_make_seq(10, 14, 1);
    indexs[1].idx.seq = af_make_seq(0, 9, 1);
    indexs[0].isSeq = true;
    indexs[1].isSeq = true;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&lhsArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&rhsArray, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_assign_gen(&outArray, lhsArray, 2, indexs, rhsArray));

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(rhsArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(lhsArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TEST(GeneralAssign, AAAA)
{
    vector<af::dim4>        numDims;
    vector< vector<float> >      in;
    vector< vector<float> >   tests;

    readTestsFromFile<float, float>(string(TEST_DIR"/gen_assign/aaaa.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af::dim4 dims2     = numDims[2];
    af::dim4 dims3     = numDims[3];
    af::dim4 dims4     = numDims[4];
    af::dim4 dims5     = numDims[5];
    af_array outArray  = 0;
    af_array rhsArray  = 0;
    af_array lhsArray  = 0;
    af_array idxArray0 = 0;
    af_array idxArray1 = 0;
    af_array idxArray2 = 0;
    af_array idxArray3 = 0;

    af_index_t indexs[4];
    indexs[0].isSeq = false;
    indexs[1].isSeq = false;
    indexs[2].isSeq = false;
    indexs[3].isSeq = false;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&lhsArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&rhsArray, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray0, &(in[2].front()),
                dims2.ndims(), dims2.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexs[0].idx.arr = idxArray0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray1, &(in[3].front()),
                dims3.ndims(), dims3.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexs[1].idx.arr = idxArray1;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray2, &(in[4].front()),
                dims4.ndims(), dims4.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexs[2].idx.arr = idxArray2;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray3, &(in[5].front()),
                dims5.ndims(), dims5.get(), (af_dtype)af::dtype_traits<float>::af_type));
    indexs[3].idx.arr = idxArray3;

    ASSERT_EQ(AF_SUCCESS, af_assign_gen(&outArray, lhsArray, 4, indexs, rhsArray));

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(rhsArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(lhsArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}


TEST(ArrayAssign, CPP_ASSIGN_INDEX)
{
    using af::array;

    const int num = 20000;

    array a = af::randu(num);
    float *hAO = a.host<float>();

    array a_copy = a;
    array idx = where(a < 0.5);
    const int len = idx.elements();
    array b = af::randu(len);
    a(idx) = b;

    float *hA = a.host<float>();
    float *hB = b.host<float>();
    float *hAC = a_copy.host<float>();
    uint *hIdx = idx.host<uint>();

    for (int i = 0; i < num; i++) {

        int j = 0;
        while(j < len) {

            // If index found, value should match B
            if ((int)hIdx[j] == i) {
                ASSERT_EQ(hA[i], hB[j]);
                break;
            }
            j++;
        }

        // If index not found, value should match original
        if (j >= len) {
            ASSERT_EQ(hA[i], hAO[i]);
        }
    }

    // hAC should not be modified, i.e. same as original
    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hAO[i], hAC[i]);
    }

    delete[] hA;
    delete[] hB;
    delete[] hAC;
    delete[] hAO;
    delete[] hIdx;
}

TEST(ArrayAssign, CPP_ASSIGN_INDEX_LOGICAL)
{
    try {
        using af::array;

        const int num = 20000;

        array a = af::randu(num);
        float *hAO = a.host<float>();

        array a_copy = a;
        array idx = where(a < 0.5);
        const int len = idx.elements();
        array b = af::randu(len);
        a(a < 0.5) = b;

        float *hA = a.host<float>();
        float *hB = b.host<float>();
        float *hAC = a_copy.host<float>();
        uint *hIdx = idx.host<uint>();

        for (int i = 0; i < num; i++) {

            int j = 0;
            while(j < len) {

                // If index found, value should match B
                if ((int)hIdx[j] == i) {
                    ASSERT_EQ(hA[i], hB[j]);
                    break;
                }
                j++;
            }

            // If index not found, value should match original
            if (j >= len) {
                ASSERT_EQ(hA[i], hAO[i]);
            }
        }

        // hAC should not be modified, i.e. same as original
        for (int i = 0; i < num; i++) {
            ASSERT_EQ(hAO[i], hAC[i]);
        }

        delete[] hA;
        delete[] hB;
        delete[] hAC;
        delete[] hAO;
        delete[] hIdx;
    } catch(af::exception &ex) {
        FAIL() << ex.what() << std::endl;
    }
}


TEST(GeneralAssign, CPP_ASNN)
{
    using namespace af;
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a = randu(nx, ny);
    array idx = where(randu(ny) > 0.5);

    const int nyb = (en - st) + 1;
    const int nxb = idx.elements();

    array b = randu(nxb, nyb);

    a(idx, seq(st, en)) = b;

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

TEST(GeneralAssign, CPP_SANN)
{
    using namespace af;
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a = randu(nx, ny);
    array idx = where(randu(ny) > 0.5);

    const int nxb = (en - st) + 1;
    const int nyb = idx.elements();

    array b = randu(nxb, nyb);

    a(seq(st, en), idx) = b;

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

TEST(GeneralAssign, CPP_SSAN)
{
    using namespace af;
    const int nx = 100;
    const int ny = 100;
    const int nz = 100;
    const int st = 20;
    const int en = 85;

    array a = randu(nx, ny, nz);
    array idx = where(randu(nz) > 0.5);

    const int nxb = (en - st) + 1;
    const int nyb = ny;
    const int nzb = idx.elements();
    array b = randu(nxb, nyb, nzb);

    a(seq(st, en), span, idx) = b;

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

TEST(GeneralAssign, CPP_AANN)
{
    using namespace af;
    const int nx = 1000;
    const int ny = 1000;

    array a = randu(nx, ny);
    array idx0 = where(randu(nx) > 0.5);
    array idx1 = where(randu(ny) > 0.5);

    const int nxb = idx0.elements();
    const int nyb = idx1.elements();
    array b = randu(nxb, nyb);

    a(idx0, idx1) = b;

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
