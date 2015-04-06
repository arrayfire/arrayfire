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
TEST(QR, CPP)
{
    if (noDoubleTests<float>()) return;

    int resultIdx = 0;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, float>(string(TEST_DIR"/lapack/qr.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array output, tau;
    af::qr(output, tau, input);

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

TEST(QRFactorized, CPP)
{
    if (noDoubleTests<float>()) return;

    int resultIdx = 0;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, float>(string(TEST_DIR"/lapack/qrfactorized.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array q, r, tau;
    af::qr(q, r, tau, input);

    af::dim4 qdims = q.dims();
    af::dim4 rdims = r.dims();

    // Get result
    float* qData = new float[qdims.elements()];
    q.host((void*)qData);
    float* rData = new float[rdims.elements()];
    r.host((void*)rData);

    // Compare result
    for (int y = 0; y < qdims[1]; ++y) {
        for (int x = 0; x < qdims[0]; ++x) {
            int elIter = y * qdims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], qData[elIter], 0.001) << "at: " << elIter << std::endl;
        }
    }

    resultIdx = 1;

    for (int y = 0; y < rdims[1]; ++y) {
        for (int x = 0; x < rdims[0]; ++x) {
            // Test only upper half
            if(x <= y) {
                int elIter = y * rdims[0] + x;
                ASSERT_NEAR(tests[resultIdx][elIter], rData[elIter], 0.001) << "at: " << elIter << std::endl;
            }
        }
    }

    // Delete
    delete[] qData;
    delete[] rData;
}
