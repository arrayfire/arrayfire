/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <af/array.h>
#include <af/arith.h>
#include <af/data.h>
#include <testHelpers.hpp>

using namespace std;
using namespace af;

TEST(JIT, CPP_JIT_HASH)
{
    using af::array;

    const int num = 20;
    const float valA = 3;
    const float valB = 5;
    const float valC = 2;
    const float valD = valA + valB;
    const float valE = valA + valC;
    const float valF1 = valD * valE - valE;
    const float valF2 = valD * valE - valD;

    array a = af::constant(valA, num);
    array b = af::constant(valB, num);
    array c = af::constant(valC, num);
    af::eval(a);
    af::eval(b);
    af::eval(c);


    // Creating a kernel
    {
        array d = a + b;
        array e = a + c;
        array f1 = d * e - e;
        float *hF1 = f1.host<float>();

        for (int i = 0; i < num; i++) {
            ASSERT_EQ(hF1[i], valF1);
        }

        delete[] hF1;
    }

    // Making sure a different kernel is generated
    {
        array d = a + b;
        array e = a + c;
        array f2 = d * e - d;
        float *hF2 = f2.host<float>();

        for (int i = 0; i < num; i++) {
            ASSERT_EQ(hF2[i], valF2);
        }

        delete[] hF2;
    }
}
