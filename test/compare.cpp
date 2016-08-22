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

template<typename T>
class Compare : public ::testing::Test
{
};

typedef ::testing::Types<float, double, uint, int, intl, uintl, uchar, short, ushort> TestTypes;
TYPED_TEST_CASE(Compare, TestTypes);

#define COMPARE(OP, Name)                                       \
    TYPED_TEST(Compare, Test_##Name)                            \
    {                                                           \
        typedef TypeParam T;                                    \
        if (noDoubleTests<T>()) return;                         \
        const int num = 1 << 20;                                \
        af_dtype ty = (af_dtype) af::dtype_traits<T>::af_type;  \
        af::array a = af::randu(num, ty);                       \
        af::array b = af::randu(num, ty);                       \
        af::array c = a OP b;                                   \
        std::vector<T> ha(num), hb(num);                        \
        std::vector<char> hc(num);                              \
        a.host(&ha[0]);                                         \
        b.host(&hb[0]);                                         \
        c.host(&hc[0]);                                         \
        for (int i = 0; i < num; i++) {                         \
            char res = ha[i] OP hb[i];                          \
            ASSERT_EQ((int)res, (int)hc[i]);                    \
        }                                                       \
    }                                                           \

COMPARE(==, eq)
COMPARE(!=, ne)
COMPARE(<=, le)
COMPARE(>=, ge)
COMPARE(<, lt)
COMPARE(>, gt)
