/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <half.hpp>
#include <testHelpers.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/random.h>

using af::array;
using af::dtype_traits;
using af::randu;
using std::vector;

template<typename T>
class Compare : public ::testing::Test {};

typedef ::testing::Types<float, double, uint, int, intl, uintl, uchar, short,
                         ushort, half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Compare, TestTypes);

#define COMPARE(OP, Name)                                   \
    TYPED_TEST(Compare, Test_##Name) {                      \
        typedef TypeParam T;                                \
        SUPPORTED_TYPE_CHECK(T);                            \
        const int num = 1 << 20;                            \
        af_dtype ty   = (af_dtype)dtype_traits<T>::af_type; \
        array a       = randu(num, ty);                     \
        array b       = randu(num, ty);                     \
        array c       = a OP b;                             \
        vector<T> ha(num), hb(num);                         \
        vector<char> hc(num);                               \
        a.host(&ha[0]);                                     \
        b.host(&hb[0]);                                     \
        c.host(&hc[0]);                                     \
        for (int i = 0; i < num; i++) {                     \
            char res = ha[i] OP hb[i];                      \
            ASSERT_EQ((int)res, (int)hc[i]);                \
        }                                                   \
    }

COMPARE(==, eq)
COMPARE(!=, ne)
COMPARE(<=, le)
COMPARE(>=, ge)
COMPARE(<, lt)
COMPARE(>, gt)
