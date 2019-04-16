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
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>

#include <af/device.h>

using af::dim4;
using af::dtype_traits;
using af::getDevice;
using af::info;
using af::setDevice;
using std::string;
using std::vector;

template<typename T>
void testFunction() {
    info();

    af_array outArray = 0;
    dim4 dims(32, 32, 1, 1);
    ASSERT_SUCCESS(af_randu(&outArray, dims.ndims(), dims.get(),
                            (af_dtype)dtype_traits<T>::af_type));
    // cleanup
    if (outArray != 0) { ASSERT_SUCCESS(af_release_array(outArray)); }
}

void infoTest() {
    int nDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&nDevices));
    ASSERT_EQ(true, nDevices > 0);

    const char* ENV = getenv("AF_MULTI_GPU_TESTS");
    if (ENV && ENV[0] == '0') {
        testFunction<float>();
    } else {
        int oldDevice = getDevice();
        for (int d = 0; d < nDevices; d++) {
            setDevice(d);
            testFunction<float>();
        }
        setDevice(oldDevice);
    }
}

TEST(Info, All) { infoTest(); }
