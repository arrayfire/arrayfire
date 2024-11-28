/*******************************************************
 * Copyright (c) 2023, ArrayFire
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

#include <af/oneapi.h>

using af::dim4;
using af::dtype_traits;
using af::getDevice;
using af::setDevice;
using std::string;
using std::vector;

using af::array;

TEST(SyclInterop, Smoke) {
    int n_devices_og = af::getDeviceCount();

    sycl::queue newque;
    afoneapi::addQueue(newque);
    EXPECT_GT(af::getDeviceCount(), n_devices_og);

    sycl::queue que   = afoneapi::getQueue();
    sycl::device dev  = afoneapi::getDevice();
    sycl::context ctx = afoneapi::getContext();
    EXPECT_FALSE(newque.get_context() == ctx);
    EXPECT_FALSE(newque.get_device() == dev);
    EXPECT_NE(&que, &newque);
    EXPECT_FALSE(que == newque);

    af::setDevice(n_devices_og);

    que = afoneapi::getQueue();
    EXPECT_NE(&que, &newque);
    EXPECT_TRUE(que == newque);

    dev = afoneapi::getDevice();
    ctx = afoneapi::getContext();
    EXPECT_TRUE(newque.get_context() == ctx);
    EXPECT_TRUE(newque.get_device() == dev);

    sycl::info::device_type dev_type = afoneapi::getDeviceType();
    sycl::platform platform          = afoneapi::getPlatform();
    EXPECT_TRUE(
        dev_type ==
        newque.get_device().get_info<sycl::info::device::device_type>());
    EXPECT_TRUE(platform == newque.get_device().get_platform());
}
