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
#include <string>
#include <vector>

using af::array;
using af::dim4;
using std::endl;
using std::string;
using std::vector;

TEST(ycbcr_rgb, InvalidArray) {
    vector<float> in(100, 1);

    dim4 dims(100);
    array input(dims, &(in.front()));

    try {
        array output = hsv2rgb(input);
        ASSERT_EQ(true, false);
    } catch (const af::exception &ex) {
        ASSERT_EQ(true, true);
        return;
    }
}

TEST(ycbcr2rgb, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(
        string(TEST_DIR "/ycbcr_rgb/ycbcr2rgb.test"), numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output = ycbcr2rgb(input);

    vector<float> outData(dims.elements());
    output.host(outData.data());

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)
            << "at: " << elIter << endl;
    }
}

TEST(ycbcr2rgb, MaxDim) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(
        string(TEST_DIR "/ycbcr_rgb/ycbcr2rgb.test"), numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));

    const size_t largeDim = 65535 * 16 + 1;
    unsigned int ntile    = (largeDim + dims[1] - 1) / dims[1];
    input                 = tile(input, 1, ntile);
    array output          = ycbcr2rgb(input);
    dim4 outDims          = output.dims();

    float *outData = new float[outDims.elements()];
    output.host((void *)outData);

    vector<float> currGoldBar = tests[0];
    for (int z = 0; z < outDims[2]; ++z) {
        for (int y = 0; y < outDims[1]; ++y) {
            for (int x = 0; x < outDims[0]; ++x) {
                int outIter =
                    (z * outDims[1] * outDims[0]) + (y * outDims[0]) + x;
                int goldIter =
                    (z * dims[1] * dims[0]) + ((y % dims[1]) * dims[0]) + x;
                ASSERT_NEAR(currGoldBar[goldIter], outData[outIter], 1.0e-3)
                    << "at: " << outIter << endl;
            }
        }
    }

    // cleanup
    delete[] outData;
}

TEST(rgb2ycbcr, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(
        string(TEST_DIR "/ycbcr_rgb/rgb2ycbcr.test"), numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output = rgb2ycbcr(input);

    vector<float> outData(dims.elements());
    output.host(outData.data());

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)
            << "at: " << elIter << endl;
    }
}

TEST(rgb2ycbcr, MaxDim) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(
        string(TEST_DIR "/ycbcr_rgb/rgb2ycbcr.test"), numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));

    const size_t largeDim = 65535 * 16 + 1;
    unsigned int ntile    = (largeDim + dims[1] - 1) / dims[1];
    input                 = tile(input, 1, ntile);
    array output          = rgb2ycbcr(input);
    dim4 outDims          = output.dims();

    float *outData = new float[outDims.elements()];
    output.host((void *)outData);

    vector<float> currGoldBar = tests[0];
    for (int z = 0; z < outDims[2]; ++z) {
        for (int y = 0; y < outDims[1]; ++y) {
            for (int x = 0; x < outDims[0]; ++x) {
                int outIter =
                    (z * outDims[1] * outDims[0]) + (y * outDims[0]) + x;
                int goldIter =
                    (z * dims[1] * dims[0]) + ((y % dims[1]) * dims[0]) + x;
                ASSERT_NEAR(currGoldBar[goldIter], outData[outIter], 1.0e-3)
                    << "at: " << outIter << endl;
            }
        }
    }

    delete[] outData;
}
