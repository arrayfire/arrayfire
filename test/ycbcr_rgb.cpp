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

TEST(ycbcr_rgb, InvalidArray)
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

TEST(ycbcr2rgb, CPP)
{
    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTestsFromFile<float,float>(string(TEST_DIR "/ycbcr_rgb/ycbcr2rgb.test"), numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af::array input(dims, &(in[0].front()));
    af::array output = af::ycbcr2rgb(input);

    float *outData = new float[dims.elements()];
    output.host((void*)outData);

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
}

TEST(rgb2ycbcr, CPP)
{
    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTestsFromFile<float,float>(string(TEST_DIR "/ycbcr_rgb/rgb2ycbcr.test"), numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af::array input(dims, &(in[0].front()));
    af::array output = af::rgb2ycbcr(input);

    float *outData = new float[dims.elements()];
    output.host((void*)outData);

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
}
