/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Include functions that provide information about the system and shouldn't
// throw exceptions during runtime.

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

TEST(NoDevice, Info) { ASSERT_SUCCESS(af_info()); }

TEST(NoDevice, InfoCxx) { af::info(); }

TEST(NoDevice, InfoString) {
    char* str;
    ASSERT_SUCCESS(af_info_string(&str, true));
    ASSERT_SUCCESS(af_free_host((void*)str));
}

TEST(NoDevice, GetDeviceCount) {
    int device = 0;
    ASSERT_SUCCESS(af_get_device_count(&device));
}

TEST(NoDevice, GetDeviceCountCxx) { af::getDeviceCount(); }

TEST(NoDevice, GetSizeOf) {
    size_t size;
    ASSERT_SUCCESS(af_get_size_of(&size, f32));
    ASSERT_EQ(4, size);
}

TEST(NoDevice, GetSizeOfCxx) {
    size_t size = af::getSizeOf(f32);
    ASSERT_EQ(4, size);
}

TEST(NoDevice, GetBackendCount) {
    unsigned int nbackends;
    ASSERT_SUCCESS(af_get_backend_count(&nbackends));
}

TEST(NoDevice, GetBackendCountCxx) {
    unsigned int nbackends = af::getBackendCount();
    UNUSED(nbackends);
}

TEST(NoDevice, GetVersion) {
    int major = 0, minor = 0, patch = 0;

    ASSERT_SUCCESS(af_get_version(&major, &minor, &patch));

    ASSERT_EQ(AF_VERSION_MAJOR, major);
    ASSERT_EQ(AF_VERSION_MINOR, minor);
    ASSERT_EQ(AF_VERSION_PATCH, patch);
}

TEST(NoDevice, GetRevision) {
    const char* revision = af_get_revision();
    UNUSED(revision);
}
