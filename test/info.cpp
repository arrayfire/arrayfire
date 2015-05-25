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
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

#include <af/device.h>

using std::string;
using std::vector;

template<typename T>
class Info : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float> TestTypes;

// register the type list
TYPED_TEST_CASE(Info, TestTypes);

template<typename T>
void infoTest()
{
    if (noDoubleTests<T>()) return;

    int nDevices = 0;
    ASSERT_EQ(AF_SUCCESS, af_get_device_count(&nDevices));

    for(int d = 0; d < nDevices; d++) {

        af::setDevice(d);
        af::info();

        af_array outArray = 0;
        af::dim4 dims(32, 32, 1, 1);
        ASSERT_EQ(AF_SUCCESS, af_randu(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
        // cleanup
        if(outArray != 0) ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    }
}

TYPED_TEST(Info, All)
{
    infoTest<TypeParam>();
}
