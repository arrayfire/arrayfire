/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <af/vision.h>
#include <string>
#include <vector>

using af::array;
using af::convolve2;
using af::dim4;
using af::dog;
using af::dtype_traits;
using af::exception;
using af::gaussianKernel;
using af::randu;
using af::sum;

template<typename T>
class DOG : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(DOG, TestTypes);

TYPED_TEST(DOG, Basic) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dim4 iDims(512, 512, 1, 1);
    array in = constant(1, iDims, (af_dtype)dtype_traits<float>::af_type);
    /* calculate DOG using ArrayFire functions */
    array k1    = gaussianKernel(3, 3);
    array k2    = gaussianKernel(2, 2);
    array smth1 = convolve2(in, k1);
    array smth2 = convolve2(in, k2);
    array diff  = smth1 - smth2;
    /* calcuate DOG using new function */
    array out = dog(in, 3, 2);
    /* compare both the values */
    float accumErr = sum<float>(out - diff);
    EXPECT_EQ(true, accumErr < 1.0e-2);
}

TYPED_TEST(DOG, Batch) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dim4 iDims(512, 512, 3, 1);
    array in = constant(1, iDims, (af_dtype)dtype_traits<float>::af_type);
    /* calculate DOG using ArrayFire functions */
    array k1    = gaussianKernel(3, 3);
    array k2    = gaussianKernel(2, 2);
    array smth1 = convolve2(in, k1);
    array smth2 = convolve2(in, k2);
    array diff  = smth1 - smth2;
    /* calcuate DOG using new function */
    array out = dog(in, 3, 2);
    /* compare both the values */
    float accumErr = sum<float>(out - diff);
    EXPECT_EQ(true, accumErr < 1.0e-2);
}

TYPED_TEST(DOG, InvalidArray) {
    array in = randu(512);
    EXPECT_THROW(dog(in, 3, 2), exception);
}
