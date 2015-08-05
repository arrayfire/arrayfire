/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;

template<typename T>
class SAT : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, uintl, intl> TestTypes;

// register the type list
TYPED_TEST_CASE(SAT, TestTypes);

TYPED_TEST(SAT, IntegralImage)
{
    if(noDoubleTests<TypeParam>()) return;

    af::array a = af::randu(530, 671, (af_dtype)af::dtype_traits<TypeParam>::af_type);
    af::array b = af::accum(a, 0);
    af::array c = af::accum(b, 1);

    af::array s = af::sat(a);

    EXPECT_EQ(true, af::allTrue<float>(c==s));
}
