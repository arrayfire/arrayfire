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

TEST(JIT, CPP_JIT_Reset_Binary)
{
    using af::array;

    af::array a = af::constant(2, 5,5);
    af::array b = af::constant(1, 5,5);
    af::array c = a + b;
    af::array d = a - b;
    af::array e = c * d;
    e.eval();
    af::array f = c - d;
    f.eval();
    af::array g = d - c;
    g.eval();

    std::vector<float> hf(f.elements());
    std::vector<float> hg(g.elements());
    f.host(&hf[0]);
    g.host(&hg[0]);

    for (int i = 0; i < (int)f.elements(); i++) {
        ASSERT_EQ(hf[i], -hg[i]);
    }
}

TEST(JIT, CPP_JIT_Reset_Unary)
{
    using af::array;

    af::array a = af::constant(2, 5,5);
    af::array b = af::constant(1, 5,5);
    af::array c = af::sin(a);
    af::array d = af::cos(b);
    af::array e = c * d;
    e.eval();
    af::array f = c - d;
    f.eval();
    af::array g = d - c;
    g.eval();

    std::vector<float> hf(f.elements());
    std::vector<float> hg(g.elements());
    f.host(&hf[0]);
    g.host(&hg[0]);

    for (int i = 0; i < (int)f.elements(); i++) {
        ASSERT_EQ(hf[i], -hg[i]);
    }
}

TEST(JIT, CPP_Multi_linear)
{
    using af::array;

    const int num = 1 << 16;
    af::array a = af::randu(num, s32);
    af::array b = af::randu(num, s32);
    af::array x = a + b;
    af::array y = a - b;
    af::eval(x, y);

    std::vector<int> ha(num);
    std::vector<int> hb(num);
    std::vector<int> hx(num);
    std::vector<int> hy(num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ((ha[i] + hb[i]), hx[i]);
        ASSERT_EQ((ha[i] - hb[i]), hy[i]);
    }
}

TEST(JIT, CPP_strided)
{
    using af::array;

    const int num = 1024;
    af::gforSet(true);
    af::array a = af::randu(num, 1, s32);
    af::array b = af::randu(1, num, s32);
    af::array x = a + b;
    af::array y = a - b;
    af::eval(x);
    af::eval(y);
    af::gforSet(false);

    std::vector<int> ha(num);
    std::vector<int> hb(num);
    std::vector<int> hx(num * num);
    std::vector<int> hy(num * num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int j = 0; j < num; j++) {
        for (int i = 0; i < num; i++) {
            ASSERT_EQ((ha[i] + hb[j]), hx[j*num + i]);
            ASSERT_EQ((ha[i] - hb[j]), hy[j*num + i]);
        }
    }
}

TEST(JIT, CPP_Multi_strided)
{
    using af::array;

    const int num = 1024;
    af::gforSet(true);
    af::array a = af::randu(num, 1, s32);
    af::array b = af::randu(1, num, s32);
    af::array x = a + b;
    af::array y = a - b;
    af::eval(x, y);
    af::gforSet(false);

    std::vector<int> ha(num);
    std::vector<int> hb(num);
    std::vector<int> hx(num * num);
    std::vector<int> hy(num * num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int j = 0; j < num; j++) {
        for (int i = 0; i < num; i++) {
            ASSERT_EQ((ha[i] + hb[j]), hx[j*num + i]);
            ASSERT_EQ((ha[i] - hb[j]), hy[j*num + i]);
        }
    }
}

TEST(JIT, CPP_Multi_pre_eval)
{
    using af::array;

    const int num = 1 << 16;
    af::array a = af::randu(num, s32);
    af::array b = af::randu(num, s32);
    af::array x = a + b;
    af::array y = a - b;

    af::eval(x);

    // Should evaluate only y
    af::eval(x, y);

    // Should not evaluate anything
    // Should not error out
    af::eval(x, y);

    std::vector<int> ha(num);
    std::vector<int> hb(num);
    std::vector<int> hx(num);
    std::vector<int> hy(num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ((ha[i] + hb[i]), hx[i]);
        ASSERT_EQ((ha[i] - hb[i]), hy[i]);
    }
}

TEST(JIT, CPP_common_node)
{
    af::array r = seq(-3, 3, 0.5);

    int n = r.dims(0);

    af::array x = af::tile(r, 1, r.dims(0));
    af::array y = af::tile(r.T(), r.dims(0), 1);
    x.eval();
    y.eval();


    std::vector<float> hx(x.elements());
    std::vector<float> hy(y.elements());
    std::vector<float> hr(r.elements());

    x.host(&hx[0]);
    y.host(&hy[0]);
    r.host(&hr[0]);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(hx[j * n + i], hr[i]);
            ASSERT_EQ(hy[j * n + i], hr[j]);
        }
    }
}

TEST(JIT, ISSUE_1646)
{
    af::array test1 = af::randn(10, 10);
    af::array test2 = af::randn(10);
    af::array test3 = af::randn(10);

    for (int i = 0; i < 1000; i++) {
        test3 += af::sum(test1, 1);
        test2 += test3;
    }
    af::eval(test2);
    af::eval(test3);
}

TEST(JIT, NonLinearLargeY)
{
    const int d0 = 2;
    // This needs to be > 2 * (1 << 20) to properly check this.
    const int d1 = 3 * (1 << 20);
    af::array a = af::randn(d0);
    af::array b = af::randn(1, d1);

    // tile is jit-ted for both the operations
    af::array c = af::tile(a, 1, d1) + af::tile(b, d0, 1);
    af::eval(c);

    std::vector<float> ha(d0);
    std::vector<float> hb(d1);
    std::vector<float> hc(d0 * d1);

    a.host(ha.data());
    b.host(hb.data());
    c.host(hc.data());

    for (int j = 0; j < d1; j++) {
        for (int i = 0; i < d0; i++) {
            ASSERT_EQ(hc[i + j * d0], ha[i] + hb[j]) << " at " << i << " , " << j;
        }
    }
}

TEST(JIT, NonLinearLargeX)
{
    af_array r, c, s;
    dim_t rdims[] = {1024000, 1, 3};
    dim_t cdims[] = {1, 1, 3};
    dim_t sdims[] = {1, 1, 1};
    dim_t ndims = 3;

    ASSERT_EQ(AF_SUCCESS, af_randu(&r, ndims, rdims, f32));
    ASSERT_EQ(AF_SUCCESS, af_constant(&c, 1, ndims, cdims, f32));
    ASSERT_EQ(AF_SUCCESS, af_eval(c));
    ASSERT_EQ(AF_SUCCESS, af_sub(&s, r, c, true));
    ASSERT_EQ(AF_SUCCESS, af_eval(s));

    dim_t relem = 1;
    dim_t celem = 1;
    dim_t selem = 1;
    for (int i = 0; i < ndims; i++) {
        relem *= rdims[i];
        celem *= cdims[i];
        sdims[i] = std::max(rdims[i], cdims[i]);
        selem *= sdims[i];
    }

    std::vector<float> hr(relem);
    std::vector<float> hc(celem);
    std::vector<float> hs(selem);

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr(hr.data(), r));
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr(hc.data(), c));
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr(hs.data(), s));

    for (int k = 0; k < sdims[2]; k++) {
        for (int j = 0; j < sdims[1]; j++) {
            for (int i = 0; i < sdims[0]; i++) {

                int sidx = i +
                    j * sdims[0] +
                    k * (sdims[0] * sdims[1]);

                int ridx = (i % rdims[0]) +
                    (j % rdims[1]) * rdims[0] +
                    (k % rdims[2]) * rdims[0] * rdims[1];

                int cidx = (i % cdims[0]) +
                    (j % cdims[1]) * cdims[0] +
                    (k % cdims[2]) * cdims[0] * cdims[1];

                ASSERT_EQ(hs[sidx], hr[ridx] - hc[cidx]) << " at " << i << "," << k;
            }
        }
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(r));
    ASSERT_EQ(AF_SUCCESS, af_release_array(c));
    ASSERT_EQ(AF_SUCCESS, af_release_array(s));
}

TEST(JIT, ISSUE_1894)
{
    af::array a = af::randu(1);
    af::array b = af::tile(a, 2 * (1 << 20));
    af::eval(b);
    float ha = -100;
    std::vector<float> hb(b.elements(), -200);

    a.host(&ha);
    b.host(hb.data());

    for (size_t i = 0; i < hb.size(); i++) {
        ASSERT_EQ(ha, hb[i]);
    }
}

TEST(JIT, LinearLarge)
{
    // Needs to be larger than 65535 * 256 (or 1 << 24)
    float v1 = std::rand() % 100;
    float v2 = std::rand() % 100;

    af::array a = af::constant(v1, 1 << 25);
    af::array b = af::constant(v2, 1 << 25);
    af::array c = (a + b) * (a - b);
    af::eval(c);

    float v3 = (v1 + v2) * (v1 - v2);

    std::vector<float> hc(c.elements());
    c.host(hc.data());

    for (size_t i = 0; i < hc.size(); i++) {
        ASSERT_EQ(hc[i], v3);
    }
}

TEST(JIT, NonLinearBuffers1)
{
    af::array a = af::randu(5, 5);
    af::array a0 = a;
    for (int i = 0; i < 1000; i++) {
        af::array b = af::randu(1, 5);
        a += af::tile(b, 5);
    }
    a.eval();
}

TEST(JIT, NonLinearBuffers2)
{
    af::array a = af::randu(100, 310);
    af::array b = af::randu(10, 10);
    for (int i = 0; i < 300; i++) {
        b += a(seq(10), seq(i, i+9)) * randu(10, 10);
    }
    b.eval();
}

TEST(JIT, TransposeBuffers)
{
    const int num = 10;
    af::array a = af::randu(1, num);
    af::array b = af::randu(1, num);
    af::array c =  a + b;
    af::array d = a.T() + b.T();

    std::vector<float> ha(a.elements());
    a.host(ha.data());

    std::vector<float> hb(b.elements());
    b.host(hb.data());

    std::vector<float> hc(c.elements());
    c.host(hc.data());

    std::vector<float> hd(d.elements());
    d.host(hd.data());

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(ha[i] + hb[i], hc[i]);
        ASSERT_FLOAT_EQ(hc[i], hd[i]);
    }
}
