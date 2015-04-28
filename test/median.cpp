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

template<typename To, typename Ti, bool flat>
void median0(int nx, int ny=1, int nz=1, int nw=1)
{
    if (noDoubleTests<Ti>()) return;
    array a = randu(nx, ny, nz, nw, (af::dtype)dtype_traits<Ti>::af_type);
    array sa = sort(a);

    Ti *h_sa = sa.host<Ti>();

    To *h_b = NULL;
    To val = 0;

    if (flat) {
        val = median<To>(a);
        h_b = &val;
    } else {
        array b = median(a);
        h_b = b.host<To>();
    }

    for (int w = 0; w < nw; w++) {
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {

                int off = (y  + ny * (z + nz * w));
                int id = nx / 2;

                if (nx & 2) {
                    ASSERT_EQ(h_sa[id + off * nx], h_b[off]);
                } else {
                    To left = h_sa[id + off * nx - 1];
                    To right = h_sa[id + off * nx];

                    ASSERT_NEAR((left + right) / 2, h_b[off], 1e-8);
                }
            }
        }
    }

    delete[] h_sa;
    if (!flat) delete[] h_b;
}

#define MEDIAN0(To, Ti)                         \
    TEST(median0, Ti##_1D_even)                 \
    {                                           \
        median0<To, Ti, false>(1000);           \
    }                                           \
    TEST(median0, Ti##_2D_even)                 \
    {                                           \
        median0<To, Ti, false>(1000, 100);      \
    }                                           \
    TEST(median0, Ti##_3D_even)                 \
    {                                           \
        median0<To, Ti, false>(1000, 25, 4);    \
    }                                           \
    TEST(median0, Ti##_4D_even)                 \
    {                                           \
        median0<To, Ti, false>(1000, 25, 2, 2); \
    }                                           \
    TEST(median0, Ti##_flat_even)               \
    {                                           \
        median0<To, Ti, true>(1000);            \
    }                                           \
    TEST(median0, Ti##_1D_odd)                  \
    {                                           \
        median0<To, Ti, false>(783);            \
    }                                           \
    TEST(median0, Ti##_2D_odd)                  \
    {                                           \
        median0<To, Ti, false>(783, 100);       \
    }                                           \
    TEST(median0, Ti##_3D_odd)                  \
    {                                           \
        median0<To, Ti, false>(783, 25, 4);     \
    }                                           \
    TEST(median0, Ti##_4D_odd)                  \
    {                                           \
        median0<To, Ti, false>(783, 25, 2, 2);  \
    }                                           \
    TEST(median0, Ti##_flat_odd)                \
    {                                           \
        median0<To, Ti, true>(783);             \
    }                                           \


MEDIAN0(float, float)
MEDIAN0(float, int)
MEDIAN0(float, uint)
MEDIAN0(float, uchar)
MEDIAN0(double, double)
