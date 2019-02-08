/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

using af::array;
using af::dim4;
using af::randomEngine;
using af::randu;
using af::setBackend;
using std::vector;

template<typename T>
class Philox : public ::testing::Test {};

// Test if CPU and GPU randu philox outputs match for the same seed
// Issue #2429
// Must run on unified backend
TEST(Philox, CpuMatchesCuda) {
    setBackend(AF_BACKEND_CUDA);

    dim4 dims_small(10);
    dim4 dims_medium(1024);
    dim4 dims_large(10000);
    dim4 dims_2D(10, 5);
    dim4 dims_3D(5, 4, 3);
    dim4 dims_4D(5, 4, 3, 2);

    int seed_small = 12;
    int seed_medium = 34;
    int seed_large = 1234;
    int seed_2D = 4321;
    int seed_3D = 56;
    int seed_4D = 78;

    setSeed(seed_small);
    array cuda_randu_small = randu(dims_small);
    setSeed(seed_medium);
    array cuda_randu_medium = randu(dims_medium);
    setSeed(seed_large);
    array cuda_randu_large = randu(dims_large);
    setSeed(seed_2D);
    array cuda_randu_2D = randu(dims_2D);
    setSeed(seed_3D);
    array cuda_randu_3D = randu(dims_3D);
    setSeed(seed_4D);
    array cuda_randu_4D = randu(dims_4D);

    vector<float> h_cuda_randu_small(cuda_randu_small.elements());
    vector<float> h_cuda_randu_medium(cuda_randu_medium.elements());
    vector<float> h_cuda_randu_large(cuda_randu_large.elements());
    vector<float> h_cuda_randu_2D(cuda_randu_2D.elements());
    vector<float> h_cuda_randu_3D(cuda_randu_3D.elements());
    vector<float> h_cuda_randu_4D(cuda_randu_4D.elements());

    cuda_randu_small.host(&h_cuda_randu_small.front());
    cuda_randu_medium.host(&h_cuda_randu_medium.front());
    cuda_randu_large.host(&h_cuda_randu_large.front());
    cuda_randu_2D.host(&h_cuda_randu_2D.front());
    cuda_randu_3D.host(&h_cuda_randu_3D.front());
    cuda_randu_4D.host(&h_cuda_randu_4D.front());

    setBackend(AF_BACKEND_CPU);

    setSeed(seed_small);
    array cpu_randu_small = randu(dims_small);
    setSeed(seed_medium);
    array cpu_randu_medium = randu(dims_medium);
    setSeed(seed_large);
    array cpu_randu_large = randu(dims_large);
    setSeed(seed_2D);
    array cpu_randu_2D = randu(dims_2D);
    setSeed(seed_3D);
    array cpu_randu_3D = randu(dims_3D);
    setSeed(seed_4D);
    array cpu_randu_4D = randu(dims_4D);

    ASSERT_VEC_ARRAY_EQ(h_cuda_randu_small, dims_small, cpu_randu_small);
    ASSERT_VEC_ARRAY_EQ(h_cuda_randu_medium, dims_medium, cpu_randu_medium);
    ASSERT_VEC_ARRAY_EQ(h_cuda_randu_large, dims_large, cpu_randu_large);
    ASSERT_VEC_ARRAY_EQ(h_cuda_randu_2D, dims_2D, cpu_randu_2D);
    ASSERT_VEC_ARRAY_EQ(h_cuda_randu_3D, dims_3D, cpu_randu_3D);
    ASSERT_VEC_ARRAY_EQ(h_cuda_randu_4D, dims_4D, cpu_randu_4D);
}
