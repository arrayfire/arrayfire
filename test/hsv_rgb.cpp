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
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;

TEST(hsv_rgb, InvalidArray)
{
    vector<float> in(100, 1);

    af::dim4 dims(100);
    af::array input(dims, &(in.front()));

    try {
        af::array output = af::hsv2rgb(input);
        ASSERT_EQ(true, false);
    } catch(af::exception) {
        ASSERT_EQ(true, true);
        return;
    }
}

TEST(hsv2rgb, CPP)
{
    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTestsFromFile<float,float>(string(TEST_DIR"/hsv_rgb/hsv2rgb.test"), numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af::array input(dims, &(in[0].front()));
    af::array output = af::hsv2rgb(input);

    std::vector<float> outData(dims.elements());
    output.host((void*)outData.data());

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)<< "at: " << elIter<< std::endl;
    }
}

TEST(rgb2hsv, CPP)
{
    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTestsFromFile<float,float>(string(TEST_DIR"/hsv_rgb/rgb2hsv.test"), numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af::array input(dims, &(in[0].front()));
    af::array output = af::rgb2hsv(input);

    std::vector<float> outData(dims.elements());
    output.host((void*)outData.data());

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)<< "at: " << elIter<< std::endl;
    }
}

TEST(rgb2hsv, MaxDim)
{
    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTestsFromFile<float,float>(string(TEST_DIR"/hsv_rgb/rgb2hsv.test"), numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af::array input(dims, &(in[0].front()));

    const size_t largeDim = 65535 * 16 + 1;
    unsigned int ntile = (largeDim + dims[1] - 1)/dims[1];
    input = af::tile(input, 1, ntile);
    af::array output = af::rgb2hsv(input);
    af::dim4 outDims = output.dims();

    float *outData = new float[outDims.elements()];
    output.host((void*)outData);

    vector<float> currGoldBar = tests[0];
    for(int z=0; z<outDims[2]; ++z) {
        for(int y=0; y<outDims[1]; ++y) {
            for(int x=0; x<outDims[0]; ++x) {
                int outIter  = (z*outDims[1]*outDims[0]) + (y*outDims[0]) + x;
                int goldIter = (z*dims[1]*dims[0]) + ((y%dims[1])*dims[0]) + x;
                ASSERT_NEAR(currGoldBar[goldIter], outData[outIter], 1.0e-3)<< "at: " << outIter << std::endl;
            }
        }
    }

    // cleanup
    delete[] outData;
}

TEST(hsv2rgb, MaxDim)
{
    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTestsFromFile<float,float>(string(TEST_DIR"/hsv_rgb/hsv2rgb.test"), numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af::array input(dims, &(in[0].front()));

    const size_t largeDim = 65535 * 16 + 1;
    unsigned int ntile = (largeDim + dims[1] - 1)/dims[1];
    input = af::tile(input, 1, ntile);
    af::array output = af::hsv2rgb(input);
    af::dim4 outDims = output.dims();

    float *outData = new float[outDims.elements()];
    output.host((void*)outData);

    vector<float> currGoldBar = tests[0];
    for(int z=0; z<outDims[2]; ++z) {
        for(int y=0; y<outDims[1]; ++y) {
            for(int x=0; x<outDims[0]; ++x) {
                int outIter  = (z*outDims[1]*outDims[0]) + (y*outDims[0]) + x;
                int goldIter = (z*dims[1]*dims[0]) + ((y%dims[1])*dims[0]) + x;
                ASSERT_NEAR(currGoldBar[goldIter], outData[outIter], 1.0e-3)<< "at: " << outIter << std::endl;
            }
        }
    }

    // cleanup
    delete[] outData;
}
