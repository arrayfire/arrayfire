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
#include <algorithm>

using namespace af;


#define MINMAXOP(fn, ty)                                \
    TEST(IndexedMinMaxTests, Test_##fn##_##ty##_0)      \
    {                                                   \
        if (noDoubleTests<ty>()) return;                \
        dtype dty = (dtype)dtype_traits<ty>::af_type;   \
        const int nx = 10000;                           \
        const int ny = 100;                             \
        af::array in = randu(nx, ny, dty);              \
        af::array val, idx;                             \
        af::fn(val, idx, in, 0);                        \
                                                        \
        ty *h_in = in.host<ty>();                       \
        ty *h_in_st = h_in;                             \
        ty *h_val = val.host<ty>();                     \
        uint *h_idx = idx.host<uint>();                 \
        for (int i = 0; i < ny; i++) {                  \
            ty tmp = *std::fn##_element(h_in, h_in +nx);\
            ASSERT_EQ(tmp, h_val[i])                    \
                << "for index" << i;                    \
            ASSERT_EQ(h_in[h_idx[i]], tmp)              \
                << "for index" << i;                    \
            h_in += nx;                                 \
        }                                               \
        af_free_host(h_in_st);                          \
        af_free_host(h_val);                            \
        af_free_host(h_idx);                            \
    }                                                   \
    TEST(IndexedMinMaxTests, Test_##fn##_##ty##_1)      \
    {                                                   \
        if (noDoubleTests<ty>()) return;                \
        dtype dty = (dtype)dtype_traits<ty>::af_type;   \
        const int nx = 100;                             \
        const int ny = 100;                             \
        af::array in = randu(nx, ny, dty);              \
        af::array val, idx;                             \
        af::fn(val, idx, in, 1);                        \
                                                        \
        ty *h_in = in.host<ty>();                       \
        ty *h_val = val.host<ty>();                     \
        uint *h_idx = idx.host<uint>();                 \
        for (int i = 0; i < nx; i++) {                  \
            ty val = h_val[i];                          \
            for (int j= 0; j < ny; j++) {               \
                ty tmp = std::fn(val, h_in[j * nx + i]);\
                ASSERT_EQ(tmp, val);                    \
            }                                           \
            ASSERT_EQ(val, h_in[h_idx[i] * nx + i]);    \
        }                                               \
        af_free_host(h_in);                             \
        af_free_host(h_val);                            \
        af_free_host(h_idx);                            \
    }                                                   \
    TEST(IndexedMinMaxTests, Test_##fn##_##ty##_all)    \
    {                                                   \
        if (noDoubleTests<ty>()) return;                \
        dtype dty = (dtype)dtype_traits<ty>::af_type;   \
        const int num = 100000;                         \
        af::array in = randu(num, dty);                 \
        ty val;                                         \
        uint idx;                                       \
        af::fn<ty>(&val, &idx, in);                     \
        ty *h_in = in.host<ty>();                       \
        ty tmp = *std::fn##_element(h_in, h_in + num);  \
        ASSERT_EQ(tmp, val);                            \
        ASSERT_EQ(tmp, h_in[idx]);                      \
        af_free_host(h_in);                             \
    }                                                   \

MINMAXOP(min, float)
MINMAXOP(min, double)
MINMAXOP(min, int)
MINMAXOP(min, uint)
MINMAXOP(min, char)
MINMAXOP(min, uchar)

MINMAXOP(max, float)
MINMAXOP(max, double)
MINMAXOP(max, int)
MINMAXOP(max, uint)
MINMAXOP(max, char)
MINMAXOP(max, uchar)

TEST(ImaxAll, IndexedSmall)
{
    const int num = 1000;
    const int st = 10;
    const int en = num - 100;
    af::array a = af::randu(num);

    float b;
    unsigned idx;
    af::max<float>(&b, &idx, a(af::seq(st, en)));

    std::vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) {
        res = std::max(res, ha[i]);
    }

    ASSERT_EQ(b, res);
}

TEST(ImaxAll, IndexedBig)
{
    const int num = 100000;
    const int st = 1000;
    const int en = num - 1000;
    af::array a = af::randu(num);

    float b;
    unsigned idx;
    af::max<float>(&b, &idx, a(af::seq(st, en)));

    std::vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) {
        res = std::max(res, ha[i]);
    }

    ASSERT_EQ(b, res);
}

TEST(IReduce, BUG_FIX_1005)
{
    const int m = 64;
    const int n = 100;
    const int b = 5;

    array in = constant(0, m, n, b);
    for (int i = 0; i < b; i++) {
        array tmp = randu(m, n);
        in(span, span, i) = tmp;

        float val0, val1;
        unsigned idx0, idx1;

        min<float>(&val0, &idx0, in(span, span, i));
        min<float>(&val1, &idx1, tmp);

        ASSERT_EQ(val0, val1);
        ASSERT_EQ(idx0, idx1);
    }
}
