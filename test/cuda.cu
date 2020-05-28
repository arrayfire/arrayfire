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

using af::allocV2;
using af::freeV2;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
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
#pragma GCC diagnostic pop

TEST(Memory, AfAllocDeviceV2CUDA) {
    void *ptr;
    ASSERT_SUCCESS(af_alloc_device_v2(&ptr, sizeof(float)));

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
    ASSERT_SUCCESS(af_free_device_v2(ptr));

    ASSERT_EQ(5, host);
}

TEST(Memory, SNIPPET_AllocCUDA) {
    //! [ex_alloc_v2_cuda]

    void *ptr = allocV2(sizeof(float));

    float *dptr     = static_cast<float *>(ptr);
    float host_data = 5.0f;

    cudaError_t error = cudaSuccess;
    error = cudaMemcpy(dptr, &host_data, sizeof(float), cudaMemcpyHostToDevice);
    freeV2(ptr);

    //! [ex_alloc_v2_cuda]
    ASSERT_EQ(cudaSuccess, error);
}
