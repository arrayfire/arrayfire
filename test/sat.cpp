/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>

using af::accum;
using af::allTrue;
using af::array;
using af::dtype_traits;
using af::randu;
using af::sat;
using std::string;
using std::vector;

template<typename T>
class SAT : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, uintl, intl,
                         short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(SAT, TestTypes);

TYPED_TEST(SAT, IntegralImage) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    array a = randu(530, 671, (af_dtype)dtype_traits<TypeParam>::af_type);
    array b = accum(a, 0);
    array c = accum(b, 1);

    array s = sat(a);

    EXPECT_EQ(true, allTrue<float>(c == s));
}
