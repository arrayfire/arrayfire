/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/array.h>
#include <af/device.h>

TEST(Memory, AfAllocDeviceCUDA) {
    void *ptr;
    ASSERT_SUCCESS(af_alloc_device(&ptr, sizeof(float)));

    /// Tests to see if the pointer returned can be used by cuda functions
    float gold_val = 5;
    float *gold    = NULL;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&gold, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(gold, &gold_val, sizeof(float),
                                      cudaMemcpyHostToDevice));

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(ptr, gold, sizeof(float), cudaMemcpyDeviceToDevice));

    float host;
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(&host, ptr, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_SUCCESS(af_free_device(ptr));

    ASSERT_EQ(5, host);
}
