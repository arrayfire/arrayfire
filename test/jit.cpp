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
