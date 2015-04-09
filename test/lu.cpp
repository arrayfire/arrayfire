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

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(LU, CPP)
{
    if (noDoubleTests<float>()) return;

    int resultIdx = 0;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, float>(string(TEST_DIR"/lapack/lu.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array output, pivot;
    af::lu(output, pivot, input);

    af::dim4 odims = output.dims();

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    for (int y = 0; y < odims[1]; ++y) {
        for (int x = 0; x < odims[0]; ++x) {
            // Check only upper triangle
            if(x <= y) {
            int elIter = y * odims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], outData[elIter], 0.001) << "at: " << elIter << std::endl;
            }
        }
    }

    // Delete
    delete[] outData;
}

TEST(LUFactorized, CPP)
{
    if (noDoubleTests<float>()) return;

    int resultIdx = 0;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, float>(string(TEST_DIR"/lapack/lufactorized.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array l, u, pivot;
    af::lu(l, u, pivot, input);

    af::dim4 ldims = l.dims();
    af::dim4 udims = u.dims();

    // Get result
    float* lData = new float[ldims.elements()];
    l.host((void*)lData);
    float* uData = new float[udims.elements()];
    u.host((void*)uData);

    // Compare result
    for (int y = 0; y < ldims[1]; ++y) {
        for (int x = 0; x < ldims[0]; ++x) {
            if(x < y);
            int elIter = y * ldims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], lData[elIter], 0.001) << "at: " << elIter << std::endl;
        }
    }

    resultIdx = 1;

    for (int y = 0; y < udims[1]; ++y) {
        for (int x = 0; x < udims[0]; ++x) {
            int elIter = y * udims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], uData[elIter], 0.001) << "at: " << elIter << std::endl;
        }
    }

    // Delete
    delete[] lData;
    delete[] uData;
}
