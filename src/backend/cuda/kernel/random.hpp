/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <debug_cuda.hpp>
#include <kernel/random_curand.hpp>
#include <kernel/random_philox.hpp>

namespace cuda
{
namespace kernel
{

    static unsigned long long seed = 0;
    static unsigned philoxcounter = 0;
    static const int THREADS = 256;
    static const int BLOCKS  = 64;
    static curandState_t *states[DeviceManager::MAX_DEVICES];
    static bool is_init[DeviceManager::MAX_DEVICES] = {0};

    void setup_states()
    {
        int device = getActiveDeviceId();

        if (!is_init[device]) {
            CUDA_CHECK(cudaMalloc(&states[device], BLOCKS * THREADS * sizeof(curandState_t)));
        }

        CUDA_LAUNCH((setup_kernel), BLOCKS, THREADS, states[device], seed);
        POST_LAUNCH_CHECK();
        is_init[device] = true;
    }

    template<typename T>
    void randu(T *out, size_t elements, const af::randomType &rtype)
    {
        switch (rtype) {
            case AF_RANDOM_DEFAULT:
                {
                    int device = getActiveDeviceId();
                    int threads = THREADS;
                    int blocks  = divup(elements, THREADS);
                    if (blocks > BLOCKS) blocks = BLOCKS;
                    CUDA_LAUNCH(uniform_kernel, blocks, threads, out, states[device], elements);
                    break;
                }
            case AF_RANDOM_PHILOX:
                {
                    int threads = THREADS;
                    int elementsPerBlockIteration = THREADS*4*sizeof(unsigned)/sizeof(T);
                    int blocks = divup(elements, elementsPerBlockIteration);
                    CUDA_LAUNCH(philox_uniform_kernel, blocks, threads,
                            out, seed, philoxcounter, elementsPerBlockIteration, elements);
                    ++philoxcounter;
                    break;
                }
        }
        POST_LAUNCH_CHECK();
    }

    template<typename T>
    void randn(T *out, size_t elements)
    {
        int device = getActiveDeviceId();

        int threads = THREADS;
        int blocks  = divup(elements, THREADS);
        if (blocks > BLOCKS) blocks = BLOCKS;

        if (!states[device]) {
            CUDA_CHECK(cudaMalloc(&states[device], BLOCKS * THREADS * sizeof(curandState_t)));

            CUDA_LAUNCH(setup_kernel, BLOCKS, THREADS, states[device], seed);

            POST_LAUNCH_CHECK();
        }

        CUDA_LAUNCH(normal_kernel, blocks, threads, out, states[device], elements);

        POST_LAUNCH_CHECK();
    }
}
}
