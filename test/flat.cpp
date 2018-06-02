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

using namespace af;

TEST(FlatTests, Test_flat_1D)
{
    const int num = 10000;
    af::array in = randu(num);
    af::array out = flat(in);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(h_in[i], h_out[i]);
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlatTests, Test_flat_2D)
{
    const int nx = 200;
    const int ny = 200;
    const int num =  nx * ny;

    af::array in = randu(nx, ny);
    af::array out = flat(in);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(h_in[i], h_out[i]);
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlatTests, Test_flat_1D_index)
{
    const int num = 10000;
    const int st = 101;
    const int en = 5000;

    af::array in = randu(num);
    af::array tmp = in(seq(st, en));
    af::array out = flat(tmp);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int i = st; i <= en; i++) {
        ASSERT_EQ(h_in[i], h_out[i - st]);
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlatTests, Test_flat_2D_index0)
{
    const int nx = 200;
    const int ny = 200;
    const int st = 21;
    const int en = 180;
    const int nxo = (en - st + 1);

    af::array in = randu(nx, ny);
    af::array tmp = in(seq(st, en), span);
    af::array out = flat(tmp);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = 0; j < ny; j++) {
        const int in_off = j * nx;
        const int out_off =j * nxo;
        for (int i = st; i <= en; i++) {
            ASSERT_EQ(h_in[i + in_off], h_out[i - st + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlatTests, Test_flat_2D_index1)
{
    const int nx = 200;
    const int ny = 200;
    const int st = 21;
    const int en = 180;

    af::array in = randu(nx, ny);
    af::array tmp = in(span, seq(st, en));
    af::array out = flat(tmp);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = st; j <= en; j++) {

        const int in_off = j * nx;
        const int out_off = (j - st) * nx;

        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(h_in[i + in_off], h_out[i + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}
