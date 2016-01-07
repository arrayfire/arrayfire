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
    //FIXME : Make this cleaner
    static const int THREADS = CURAND_THREADS;
    static const int BLOCKS  = CURAND_BLOCKS;
    static unsigned philoxcounter = 0;

    void setup_states()
    {
        curandState_t *state = getcurandState();
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
                    curandState_t *state = getcurandState();
                    CUDA_LAUNCH(uniform_kernel, blocks, threads, out, state, elements);
                    POST_LAUNCH_CHECK();
                    break;
                }
            case AF_RANDOM_PHILOX:
                {
                    int threads = THREADS;
                    int elementsPerBlockIteration = THREADS*4*sizeof(unsigned)/sizeof(T);
                    int blocks = divup(elements, elementsPerBlockIteration);
                    //FIXME : Use a different seed for philox
                    CUDA_LAUNCH(philox_uniform_kernel, blocks, threads,
                            out, seeds[getActiveDeviceId()], philoxcounter, elementsPerBlockIteration, elements);
                    POST_LAUNCH_CHECK();
                    ++philoxcounter;
                    break;
                }
        }
    }

    template<typename T>
    void randn(T *out, size_t elements)
    {
        int device = getActiveDeviceId();

        int threads = THREADS;
        int blocks  = divup(elements, THREADS);
        if (blocks > BLOCKS) blocks = BLOCKS;

        curandState_t *state = getcurandState();

        CUDA_LAUNCH(normal_kernel, blocks, threads, out, state, elements);

        POST_LAUNCH_CHECK();
    }
}
}
