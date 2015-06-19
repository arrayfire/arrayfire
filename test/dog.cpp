/*******************************************************
 * Copyright (c) 2015, ArrayFire
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

template<typename T>
class DOG : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(DOG, TestTypes);


TYPED_TEST(DOG, Basic)
{
    if (noDoubleTests<TypeParam>()) return;

    af::dim4 iDims(512, 512, 1, 1);
    af::array in = af::constant<TypeParam>(1, iDims);
    /* calculate DOG using ArrayFire functions */
    af::array k1    = af::gaussianKernel(3, 3);
    af::array k2    = af::gaussianKernel(2, 2);
    af::array smth1 = af::convolve2(in, k1);
    af::array smth2 = af::convolve2(in, k2);
    af::array diff  = smth1 - smth2;
    /* calcuate DOG using new function */
    af::array out= af::dog(in, 3, 2);
    /* compare both the values */
    float accumErr = af::sum<float>(out-diff);
    EXPECT_EQ(true, accumErr<1.0e-2);
}

TYPED_TEST(DOG, Batch)
{
    if (noDoubleTests<TypeParam>()) return;

    af::dim4 iDims(512, 512, 3, 1);
    af::array in = af::constant<TypeParam>(1, iDims);
    /* calculate DOG using ArrayFire functions */
    af::array k1    = af::gaussianKernel(3, 3);
    af::array k2    = af::gaussianKernel(2, 2);
    af::array smth1 = af::convolve2(in, k1);
    af::array smth2 = af::convolve2(in, k2);
    af::array diff  = smth1 - smth2;
    /* calcuate DOG using new function */
    af::array out= af::dog(in, 3, 2);
    /* compare both the values */
    float accumErr = af::sum<float>(out-diff);
    EXPECT_EQ(true, accumErr<1.0e-2);
}

TYPED_TEST(DOG, InvalidArray)
{
    af::array in = af::randu(512);
    EXPECT_THROW(af::dog(in, 3, 2),
                 af::exception);
}
