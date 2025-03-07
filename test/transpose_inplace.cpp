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

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::endl;
using std::vector;

template<typename T>
class Transpose : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, uint, char, uchar,
                         short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Transpose, TestTypes);

template<typename T>
void transposeip_test(dim4 dims) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray  = 0;
    af_array outArray = 0;

    ASSERT_SUCCESS(af_randu(&inArray, dims.ndims(), dims.get(),
                            (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_transpose(&outArray, inArray, false));
    ASSERT_SUCCESS(af_transpose_inplace(inArray, false));

    ASSERT_ARRAYS_EQ(inArray, outArray);

    // cleanup
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

#define INIT_TEST(Side, D3, D4)                                \
    TYPED_TEST(Transpose, TranposeIP_##Side) {                 \
        transposeip_test<TypeParam>(dim4(Side, Side, D3, D4)); \
    }

INIT_TEST(10, 1, 1);
INIT_TEST(64, 1, 1);
INIT_TEST(300, 1, 1);
INIT_TEST(1000, 1, 1);
INIT_TEST(100, 2, 1);
INIT_TEST(25, 2, 2);

////////////////////////////////////// CPP //////////////////////////////////
//
void transposeInPlaceCPPTest() {
    dim4 dims(64, 64, 1, 1);

    array input  = randu(dims);
    array output = transpose(input);
    transposeInPlace(input);

    ASSERT_ARRAYS_EQ(input, output);
}
