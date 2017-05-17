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
