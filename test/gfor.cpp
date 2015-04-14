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
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;
using namespace af;

TEST(GFOR, Assign_Scalar_Span)
{
    const int num = 1000;
    const float val = 3;
    array A = randu(num);

    gfor(seq ii, num) {
        A(ii) = val;
    }

    float *hA = A.host<float>();

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hA[i], val);
    }

    delete[] hA;
}

TEST(GFOR, Assign_Scalar_Seq)
{
    const int num = 1000;
    const int st = 100;
    const int en = 500;
    const float val = 3;
    array A = randu(num);
    array B = A.copy();

    gfor(seq ii, st, en) {
        A(ii) = val;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();

    for (int i = 0; i < num; i++) {
        if (i >= st && i <= en) ASSERT_EQ(hA[i], val);
        else ASSERT_EQ(hA[i], hB[i]);
    }

    delete[] hA;
    delete[] hB;
}

TEST(GFOR, Inc_Scalar_Span)
{
    const int num = 1000;
    const float val = 3;
    array A = randu(num);
    array B = A.copy();

    gfor(seq ii, num) {
        A(ii) += val;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hA[i], val + hB[i]);
    }

    delete[] hA;
    delete[] hB;
}

TEST(GFOR, Inc_Scalar_Seq)
{
    const int num = 1000;
    const int st = 100;
    const int en = 500;
    const float val = 3;
    array A = randu(num);
    array B = A.copy();

    gfor(seq ii, st, en) {
        A(ii) += val;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();

    for (int i = 0; i < num; i++) {
        if (i >= st && i <= en) ASSERT_EQ(hA[i], hB[i] + val);
        else ASSERT_EQ(hA[i], hB[i]);
    }

    delete[] hA;
    delete[] hB;
}

TEST(GFOR, Assign_Array_Span)
{
    const int nx = 1000;
    array A = randu(nx);
    array B = randu(1, 1);

    gfor(seq ii, nx) {
        A(ii) = B;
    }

    float *hA = A.host<float>();
    float val = B.scalar<float>();

    for (int i = 0; i < nx; i++) {
        ASSERT_EQ(hA[i], val);
    }

    delete[] hA;
}

TEST(GFOR, Assign_Array_Seq)
{
    const int nx = 1000;
    const int ny = 25;
    const int st = 100;
    const int en = 500;
    array A = randu(nx, ny);
    array B = A.copy();
    array C = randu(1, ny);

    gfor(seq ii, st, en) {
        A(ii, span) = C;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int j = 0; j < ny; j++) {
        float val = hC[j];
        const int off = j * nx;
        for (int i = 0; i < nx; i++) {
            if (i >= st && i <= en) ASSERT_EQ(hA[i + off], val);
            else ASSERT_EQ(hA[i + off], hB[i + off]);
        }
    }

    delete[] hA;
    delete[] hB;
    delete[] hC;
}

TEST(GFOR, Inc_Array_Span)
{
    const int nx = 1000;
    array A = randu(nx);
    array B = A.copy();
    array C = randu(1, 1);

    gfor(seq ii, nx) {
        A(ii) += C;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float val = C.scalar<float>();

    for (int i = 0; i < nx; i++) {
        ASSERT_EQ(hA[i], val + hB[i]);
    }

    delete[] hA;
    delete[] hB;
}

TEST(GFOR, Inc_Array_Seq)
{
    const int nx = 1000;
    const int ny = 25;
    const int st = 100;
    const int en = 500;
    array A = randu(nx, ny);
    array B = A.copy();
    array C = randu(1, ny);

    gfor(seq ii, st, en) {
        A(ii, span) += C;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int j = 0; j < ny; j++) {
        float val = hC[j];
        const int off = j * nx;
        for (int i = 0; i < nx; i++) {
            if (i >= st && i <= en) ASSERT_EQ(hA[i + off], val + hB[i + off]);
            else ASSERT_EQ(hA[i + off], hB[i + off]);
        }
    }

    delete[] hA;
    delete[] hB;
    delete[] hC;
}
