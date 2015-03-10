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


#define CONSTANT_TESTS(TY, VAL)                         \
    TEST(ConstantTests, Test_##TY)                      \
    {                                                   \
        if (noDoubleTests<TY>()) return;                \
        const int num = 1000;                           \
        TY val = VAL;                                   \
        dtype dty = (dtype) dtype_traits<TY>::af_type;  \
        af::array in = constant(val, num, dty);         \
                                                        \
        TY *h_in = in.host<TY>();                       \
                                                        \
        for (int i = 0; i < num; i++) {                 \
            ASSERT_EQ(h_in[i], val);                    \
        }                                               \
                                                        \
        delete[] h_in;                                  \
    }                                                   \

CONSTANT_TESTS(float, 3.5);
CONSTANT_TESTS(double, 5.5);
CONSTANT_TESTS(int, ((1 << 31) + (1 << 24)));
CONSTANT_TESTS(unsigned, ((1u << 31) + (1u << 24)));
CONSTANT_TESTS(uchar, 255);
CONSTANT_TESTS(uintl, ((1UL << 63) + (1UL << 54)));
CONSTANT_TESTS(intl, ((1LL << 63) + (1LL << 54)));
