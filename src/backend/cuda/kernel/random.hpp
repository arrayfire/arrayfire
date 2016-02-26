/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <curand_kernel.h>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
namespace kernel
{

    static const int THREADS = 256;
    static const int BLOCKS  = 64;
    static unsigned long long seeds[DeviceManager::MAX_DEVICES] = {0};

    __global__ static void
    setup_kernel(curandState_t *states, unsigned long long seed)
    {
        unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
        curand_init(seed, tid, 0, &states[tid]);
    }

    class curandStateManager
    {
        curandState_t *_state;
        unsigned long long _seed;

        void resetSeed()
        {
            CUDA_LAUNCH(setup_kernel, BLOCKS, THREADS, _state, _seed);

            POST_LAUNCH_CHECK();
        }

        public:
        curandStateManager()
            : _state(NULL), _seed(0)
        {
        }

        ~curandStateManager()
        {
            try {
                if (_state != NULL) {
                    cudaError_t err = cudaFree(_state);
                    if (err != cudaErrorCudartUnloading) {
                        CUDA_CHECK(err);
                    }
                }
            } catch (AfError err) {
                if (err.getError() != AF_ERR_DRIVER) { // Can happen from cudaErrorDevicesUnavailable
                    throw err;
                }
            }
        }

        unsigned long long getSeed() const
        {
            return _seed;
        }

        void setSeed(const unsigned long long in_seed)
        {
            _seed = in_seed;
            this->resetSeed();
        }

        curandState_t* getState()
        {
            if(_state)
                return _state;

            CUDA_CHECK(cudaMalloc((void **)&_state, BLOCKS * THREADS * sizeof(curandState_t)));
            this->resetSeed();
            return _state;
        }
    };

    curandState_t* getcurandState()
    {
        static curandStateManager states[cuda::DeviceManager::MAX_DEVICES];

        int id = cuda::getActiveDeviceId();

        if(!(states[id].getState())) {
            // states[id] was not initialized. Very bad.
            // Throw an error here
        }

        if(states[id].getSeed() != seeds[id]) {
            states[id].setSeed(seeds[id]);
        }

        return states[id].getState();
    }

    template<typename T>
    __device__
    void generate_uniform(T *val, curandState_t *state)
    {
        *val =  (T)curand(state);
    }

    template<> __device__
    void generate_uniform<char>(char *val, curandState_t *state)
    {
        *val = curand_uniform(state) > 0.5;
    }

    template<> __device__
    void generate_uniform<float>(float *val, curandState_t *state)
    {
        *val = curand_uniform(state);
    }

    template<> __device__
    void generate_uniform<double>(double *val, curandState_t *state)
    {
        *val = curand_uniform_double(state);
    }

    template<> __device__
    void generate_uniform<cfloat>(cfloat *cval, curandState_t *state)
    {
        cval->x = curand_uniform(state);
        cval->y = curand_uniform(state);
    }

    template<> __device__
    void generate_uniform<cdouble>(cdouble *cval, curandState_t *state)
    {
        cval->x = curand_uniform_double(state);
        cval->y = curand_uniform_double(state);
    }


    __device__
    void generate_normal(float *val, curandState_t *state)
    {
        *val = curand_normal(state);
    }


    __device__
    void generate_normal(double *val, curandState_t *state)
    {
        *val = curand_normal_double(state);
    }


    __device__
    void generate_normal(cfloat *cval, curandState_t *state)
    {
        cval->x = curand_normal(state);
        cval->y = curand_normal(state);
    }


    __device__
    void generate_normal(cdouble *cval, curandState_t *state)
    {
        cval->x = curand_normal_double(state);
        cval->y = curand_normal_double(state);
    }

    template<typename T>
    __global__ static void
    uniform_kernel(T *out, curandState_t *states, size_t elements)
    {
        unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
        curandState_t state = states[id];
        for (int tid = id; tid < elements; tid += blockDim.x * gridDim.x) {
            T value;
            generate_uniform<T>(&value, &state);
            out[tid] = value;
        }
        states[id] = state;
    }

    template<typename T>
    __global__ static void
    normal_kernel(T *out, curandState_t *states, size_t elements)
    {
        unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
        curandState_t state = states[id];
        for (int tid = id; tid < elements; tid += blockDim.x * gridDim.x) {
            T value;
            generate_normal(&value, &state);
            out[tid] = value;
        }
        states[id] = state;
    }

    void setup_states()
    {
        curandState_t *state = getcurandState();
    }

    template<typename T>
    void randu(T *out, size_t elements)
    {
        int device = getActiveDeviceId();

        int threads = THREADS;
        int blocks  = divup(elements, THREADS);
        if (blocks > BLOCKS) blocks = BLOCKS;

        curandState_t *state = getcurandState();

        CUDA_LAUNCH(uniform_kernel, blocks, threads, out, state, elements);
        POST_LAUNCH_CHECK();
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
