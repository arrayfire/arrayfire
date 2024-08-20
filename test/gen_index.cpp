/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/algorithm.h>
#include <af/arith.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/random.h>
#include <af/traits.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using af::dim4;
using af::dtype_traits;
using std::endl;
using std::get;
using std::ostream_iterator;
using std::string;
using std::stringstream;
using std::vector;

struct index_test {
    string filename_;
    dim4 dims_;
    index_test(string filename, dim4 dims) : filename_(filename), dims_(dims) {}
};

using index_params = std::tuple<index_test, af_dtype, af_dtype>;

class IndexGeneralizedLegacy : public ::testing::TestWithParam<index_params> {
    void SetUp() {
        index_params params = GetParam();
        vector<dim4> numDims;
        vector<vector<float>> in;
        vector<vector<float>> tests;

        if (noDoubleTests(get<1>(params))) return;
        if (noHalfTests(get<1>(params))) return;

        if (noDoubleTests(get<2>(params))) return;
        if (noHalfTests(get<2>(params))) return;
        readTestsFromFile<float, float>(get<0>(params).filename_, numDims, in,
                                        tests);

        dim4 dims0 = numDims[0];
        dim4 dims1 = numDims[1];

        af_array inTmp = 0;
        ASSERT_SUCCESS(af_create_array(&inTmp, &(in[0].front()), dims0.ndims(),
                                       dims0.get(), f32));

        ASSERT_SUCCESS(af_cast(&inArray_, inTmp, get<1>(params)));
        af_release_array(inTmp);

        af_array idxTmp = 0;
        ASSERT_SUCCESS(af_create_array(&idxTmp, &(in[1].front()), dims1.ndims(),
                                       dims1.get(), f32));
        ASSERT_SUCCESS(af_cast(&idxArray_, idxTmp, get<2>(params)));
        af_release_array(idxTmp);

        vector<float> hgold = tests[0];
        af_array goldTmp;
        af_create_array(&goldTmp, &hgold.front(), get<0>(params).dims_.ndims(),
                        get<0>(params).dims_.get(), f32);
        ASSERT_SUCCESS(af_cast(&gold_, goldTmp, get<1>(params)));
        af_release_array(goldTmp);
    }

    void TearDown() {
        if (inArray_) { ASSERT_SUCCESS(af_release_array(inArray_)); }
        if (idxArray_) { ASSERT_SUCCESS(af_release_array(idxArray_)); }
        if (gold_) { ASSERT_SUCCESS(af_release_array(gold_)); }
    }

   public:
    IndexGeneralizedLegacy() : gold_(0), inArray_(0), idxArray_(0) {}

    af_array gold_;
    af_array inArray_;
    af_array idxArray_;
};

string testNameGenerator(
    const ::testing::TestParamInfo<IndexGeneralizedLegacy::ParamType> info) {
    stringstream ss;
    ss << "type_" << get<1>(info.param) << "_idx_type_" << get<2>(info.param);
    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    Legacy, IndexGeneralizedLegacy,
    ::testing::Combine(
        ::testing::Values(index_test(
            string(TEST_DIR "/gen_index/s0_3s0_1s1_2a.test"), dim4(4, 2, 2))),
        ::testing::Values(f32, f64, c32, c64, u64, s64, u16, s16, u8, b8, f16),
        ::testing::Values(f32, f64, u64, s64, u16, s16, u8, f16)),
    testNameGenerator);

TEST_P(IndexGeneralizedLegacy, SSSA) {
    index_params params = GetParam();
    if (noDoubleTests(get<1>(params))) return;
    if (noHalfTests(get<1>(params))) return;

    if (noDoubleTests(get<2>(params))) return;
    if (noHalfTests(get<2>(params))) return;

    af_array outArray = 0;
    af_index_t indexes[4];
    indexes[0].idx.seq = af_make_seq(0, 3, 1);
    indexes[1].idx.seq = af_make_seq(0, 1, 1);
    indexes[2].idx.seq = af_make_seq(1, 2, 1);
    indexes[3].idx.arr = idxArray_;
    indexes[0].isSeq   = true;
    indexes[1].isSeq   = true;
    indexes[2].isSeq   = true;
    indexes[3].isSeq   = false;
    ASSERT_SUCCESS(af_index_gen(&outArray, inArray_, 4, indexes));
    ASSERT_ARRAYS_EQ(gold_, outArray);
    af_release_array(outArray);
}

void testGeneralIndexOneArray(string pTestFile, const dim_t ndims,
                              af_index_t *indexs, int arrayDim) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(pTestFile, numDims, in, tests);

    dim4 dims0        = numDims[0];
    dim4 dims1        = numDims[1];
    af_array outArray = 0;
    af_array inArray  = 0;
    af_array idxArray = 0;

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_SUCCESS(af_create_array(&idxArray, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (af_dtype)dtype_traits<float>::af_type));
    indexs[arrayDim].idx.arr = idxArray;

    ASSERT_SUCCESS(af_index_gen(&outArray, inArray, ndims, indexs));

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    vector<float> outData(nElems);

    ASSERT_SUCCESS(af_get_data_ptr((void *)outData.data(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(idxArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TEST(GeneralIndex, ASSS) {
    af_index_t indexs[4];
    indexs[1].idx.seq = af_make_seq(0, 9, 1);
    indexs[2].idx.seq = af_span;
    indexs[3].idx.seq = af_span;
    indexs[0].isSeq   = false;
    indexs[1].isSeq   = true;
    indexs[2].isSeq   = true;
    indexs[3].isSeq   = true;

    testGeneralIndexOneArray(string(TEST_DIR "/gen_index/as0_9s0_ns0_n.test"),
                             4, indexs, 0);
}

TEST(GeneralIndex, SASS) {
    af_index_t indexs[2];
    indexs[0].idx.seq = af_make_seq(10, 40, 1);
    indexs[0].isSeq   = true;
    indexs[1].isSeq   = false;

    testGeneralIndexOneArray(string(TEST_DIR "/gen_index/s0_9as0_ns0_n.test"),
                             2, indexs, 1);
}

TEST(GeneralIndex, AASS) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(
        string(TEST_DIR "/gen_index/aas0_ns0_n.test"), numDims, in, tests);

    dim4 dims0         = numDims[0];
    dim4 dims1         = numDims[1];
    dim4 dims2         = numDims[2];
    af_array outArray  = 0;
    af_array inArray   = 0;
    af_array idxArray0 = 0;
    af_array idxArray1 = 0;

    af_index_t indexs[2];

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_SUCCESS(af_create_array(&idxArray0, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (af_dtype)dtype_traits<float>::af_type));
    indexs[0].isSeq   = false;
    indexs[0].idx.arr = idxArray0;

    ASSERT_SUCCESS(af_create_array(&idxArray1, &(in[2].front()), dims2.ndims(),
                                   dims2.get(),
                                   (af_dtype)dtype_traits<float>::af_type));
    indexs[1].isSeq   = false;
    indexs[1].idx.arr = idxArray1;

    ASSERT_SUCCESS(af_index_gen(&outArray, inArray, 2, indexs));

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    vector<float> outData(nElems);

    ASSERT_SUCCESS(af_get_data_ptr((void *)outData.data(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(idxArray0));
    ASSERT_SUCCESS(af_release_array(idxArray1));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TEST(GeneralIndex, SSAS_LinearSteps) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;  // Read tests from file

    readTestsFromFile<float, float>(
        TEST_DIR "/gen_index/s29_9__3s0_9_2as0_n.test", numDims, in, tests);

    af_array outArray  = 0;
    af_array inArray   = 0;
    af_array idxArray0 = 0;
    dim4 dims0         = numDims[0];
    dim4 dims1         = numDims[1];

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_SUCCESS(af_create_array(&idxArray0, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    af_index_t indexs[4];
    indexs[0].idx.seq = af_make_seq(29, 9, -3);
    indexs[1].idx.seq = af_make_seq(0, 9, 2);
    indexs[2].idx.arr = idxArray0;
    indexs[3].idx.seq = af_span;

    indexs[0].isSeq = true;
    indexs[1].isSeq = true;
    indexs[2].isSeq = false;
    indexs[3].isSeq = true;

    ASSERT_SUCCESS(af_index_gen(&outArray, inArray, 4, indexs));

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    vector<float> outData(nElems);

    ASSERT_SUCCESS(af_get_data_ptr((void *)outData.data(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

using af::array;
using af::freeHost;
using af::randu;
using af::seq;
using af::span;
using af::where;

TEST(GeneralIndex, CPP_ASNN) {
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a   = randu(nx, ny);
    array idx = where(randu(nx) > 0.5);
    array b   = a(idx, seq(st, en));

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);

    float *hA  = a.host<float>();
    uint *hIdx = idx.host<uint>();
    float *hB  = b.host<float>();

    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + (st + j) * nx;
        float *hBt = hB + j * nxb;
        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[hIdx[i]], hBt[i]) << "at " << i << " " << j << endl;
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx);
}

TEST(GeneralIndex, CPP_SANN) {
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a   = randu(nx, ny);
    array idx = where(randu(ny) > 0.5);
    array b   = a(seq(st, en), idx);

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);

    float *hA  = a.host<float>();
    uint *hIdx = idx.host<uint>();
    float *hB  = b.host<float>();

    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + hIdx[j] * nx;
        float *hBt = hB + j * nxb;

        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[i + st], hBt[i]) << "at " << i << " " << j << endl;
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx);
}

TEST(GeneralIndex, CPP_SSAN) {
    const int nx = 100;
    const int ny = 100;
    const int nz = 100;
    const int st = 20;
    const int en = 85;

    array a   = randu(nx, ny, nz);
    array idx = where(randu(nz) > 0.5);
    array b   = a(seq(st, en), span, idx);

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);
    const int nzb = b.dims(2);

    float *hA  = a.host<float>();
    uint *hIdx = idx.host<uint>();
    float *hB  = b.host<float>();

    for (int k = 0; k < nzb; k++) {
        float *hAt = hA + hIdx[k] * nx * ny;
        float *hBt = hB + k * nxb * nyb;

        for (int j = 0; j < nyb; j++) {
            for (int i = 0; i < nxb; i++) {
                ASSERT_EQ(hAt[j * nx + i + st], hBt[j * nxb + i])
                    << "at " << i << " " << j << " " << k << endl;
            }
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx);
}

TEST(GeneralIndex, CPP_AANN) {
    const int nx = 1000;
    const int ny = 1000;

    array a    = randu(nx, ny);
    array idx0 = where(randu(nx) > 0.5);
    array idx1 = where(randu(ny) > 0.5);
    array b    = a(idx0, idx1);

    const int nxb = b.dims(0);
    const int nyb = b.dims(1);

    float *hA   = a.host<float>();
    uint *hIdx0 = idx0.host<uint>();
    uint *hIdx1 = idx1.host<uint>();
    float *hB   = b.host<float>();

    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + hIdx1[j] * nx;
        float *hBt = hB + j * nxb;
        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[hIdx0[i]], hBt[i]) << "at " << i << " " << j << endl;
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx0);
    freeHost(hIdx1);
}
