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
#include <af/index.h>
#include <testHelpers.hpp>

using namespace af;

TEST(FlipTests, Test_flip_1D)
{
    const int num = 10000;
    af::array in = randu(num);
    af::array out = flip(in, 0);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(h_in[num - i - 1], h_out[i])
            << "at (" << i << ")";
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlipTests, Test_flip_2D0)
{
    const int nx = 200;
    const int ny = 200;

    af::array in = randu(nx, ny);
    af::array out = flip(in, 0);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = 0; j < ny; j++) {
        int off = j * nx;
        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(h_in[off + nx - 1 - i], h_out[off + i])
                << "at (" << i << "," << j << ")";

        }
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlipTests, Test_flip_2D1)
{
    const int nx = 200;
    const int ny = 200;

    af::array in = randu(nx, ny);
    af::array out = flip(in, 1);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = 0; j < ny; j++) {
        int ioff = (ny - 1 - j) * nx;
        int ooff = j * nx;
        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(h_in[ioff + i], h_out[ooff + i])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}


TEST(FlipTests, Test_flip_1D_index)
{
    const int num = 10000;
    const int st = 101;
    const int en = 5000;

    af::array in = randu(num);
    af::array tmp = in(seq(st, en));
    af::array out = flip(tmp, 0);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int i = st; i <= en; i++) {
        ASSERT_EQ(h_in[i], h_out[en - i])
            << "at (" << i << ")";
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlipTests, Test_flip_2D_index00)
{
    const int nx = 200;
    const int ny = 200;
    const int st = 21;
    const int en = 180;
    const int nxo = (en - st + 1);

    af::array in = randu(nx, ny);
    af::array tmp = in(seq(st, en), span);
    af::array out = flip(tmp, 0);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = 0; j < ny; j++) {
        const int in_off = j * nx;
        const int out_off =j * nxo;
        for (int i = st; i <= en; i++) {
            ASSERT_EQ(h_in[i + in_off], h_out[en - i + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlipTests, Test_flip_2D_index01)
{
    const int nx = 200;
    const int ny = 200;
    const int st = 21;
    const int en = 180;
    const int nxo = (en - st + 1);

    af::array in = randu(nx, ny);
    af::array tmp = in(seq(st, en), span);
    af::array out = flip(tmp, 1);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = 0; j < ny; j++) {
        const int in_off = (ny - 1 - j) * nx;
        const int out_off =j * nxo;
        for (int i = st; i <= en; i++) {
            ASSERT_EQ(h_in[i + in_off], h_out[i - st + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlipTests, Test_flip_2D_index10)
{
    const int nx = 200;
    const int ny = 200;
    const int st = 21;
    const int en = 180;

    af::array in = randu(nx, ny);
    af::array tmp = in(span, seq(st, en));
    af::array out = flip(tmp, 0);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = st; j <= en; j++) {

        const int in_off = j * nx;
        const int out_off = (j - st) * nx;

        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(h_in[nx - 1 - i + in_off], h_out[i + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlipTests, Test_flip_2D_index11)
{
    const int nx = 200;
    const int ny = 200;
    const int st = 21;
    const int en = 180;

    af::array in = randu(nx, ny);
    af::array tmp = in(span, seq(st, en));
    af::array out = flip(tmp, 1);

    float *h_in = in.host<float>();
    float *h_out = out.host<float>();

    for (int j = st; j <= en; j++) {

        const int in_off = j * nx;
        const int out_off = (en - j) * nx;

        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(h_in[i + in_off], h_out[i + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}
