#include <curand_kernel.h>
#include <dispatch.hpp>

namespace cuda
{
namespace kernel
{

    static const int BLOCK_DIM = 256;
    static unsigned long long uniform_seed = 0;
    static unsigned long long normal_seed  = 0;

    template<typename T>
    __device__
    void generate_uniform(T *val, curandState_t *state)
    {
        *val =  (T)curand(state);
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

    __global__ static void
    setup_kernel(curandState_t *states, unsigned long long seed, size_t elements)
    {
        unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < elements) {
            curandState_t state;
            curand_init(seed, tid, 0, &state);
            states[tid] = state;
        }
    }

    template<typename T>
    __global__ static void
    uniform_kernel(T *out, curandState_t *states, size_t elements)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < elements) {
            curandState_t state = states[tid];
            T value;
            generate_uniform<T>(&value, &state);
            out[tid] = value;
            states[tid] = state;
        }
    }

    template<typename T>
    __global__ static void
    normal_kernel(T *out, curandState_t *states, size_t elements)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < elements) {
            curandState_t state = states[tid];
            T value;
            generate_normal(&value, &state);
            out[tid] = value;
            states[tid] = state;
        }
    }

    template<typename T>
    void randu(T *out, size_t elements)
    {
        // FIXME: check for block size limit
        dim3 threads(BLOCK_DIM);
        dim3 blocks(divup(elements, BLOCK_DIM));

        static curandState_t *states = NULL;
        if (states == NULL) {
            cudaMalloc(&states, blocks.x * threads.x * sizeof(curandState_t));
            setup_kernel<<<threads, blocks>>>(states, uniform_seed, elements);
        }

        uniform_kernel<<<threads, blocks>>>(out, states, elements);
    }

    template<typename T>
    void randn(T *out, size_t elements)
    {
        // FIXME: check for block size limit
        dim3 threads(BLOCK_DIM);
        dim3 blocks(divup(elements, BLOCK_DIM));

        static curandState_t *states = NULL;
        if (states == NULL) {
            cudaMalloc(&states, blocks.x * threads.x * sizeof(curandState_t));
            setup_kernel<<<threads, blocks>>>(states, normal_seed, elements);
        }

        normal_kernel<<<threads, blocks>>>(out, states, elements);
    }
}
}
