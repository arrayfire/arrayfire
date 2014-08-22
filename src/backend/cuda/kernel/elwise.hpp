#pragma once

#include "../helper.hpp"

namespace cuda
{
namespace kernel
{

template<typename T>
class plus
{
public:
    __device__ __host__ T operator()(const T &lhs, const T &rhs) {
        return lhs+rhs;
    }
};

template<typename O, typename T, typename U, typename Op>
__global__
void binaryOpKernel(O*const  out, const T*const  lhs, const U* const rhs, const size_t elements)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < elements)
        out[idx] = Op()(lhs[idx], rhs[idx]);
}

template<typename O, typename T, typename U>
void binaryOp(O* out, const T* lhs, const U* rhs, const size_t &elements)
{
    dim3 threads(512);
    dim3 blocks(divup(elements,threads.x));

    binaryOpKernel<O,T,U,plus<O> ><<<threads, blocks>>>(out, lhs, rhs, elements);
}

template<typename T>
void set(T* ptr, T val, const size_t &elements);
}
}
