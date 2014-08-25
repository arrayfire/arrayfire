#pragma once

#include <backend.hpp>
#include <cassert>

#define divup(a, b) ((a)+(b)-1)/(b)

#define cudaCheckError() {                                              \
     cudaError_t e=cudaGetLastError();                                  \
     if(e!=cudaSuccess) {                                               \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(0);                                                    \
          }                                                             \
}

namespace cuda
{

template <typename T>
struct SharedMemory
{
    // return a pointer to the runtime-sized shared memory array.
    __device__ T* getPointer()
    {
        extern __device__ void Error_UnsupportedType(); // Ensure that we won't compile any un-specialized types
        Error_UnsupportedType();
        return (T*)0;
    }
};

#define DECLARE_SPECIALIZATIONS(T)                                          \
template <>                                                                 \
struct SharedMemory <T>                                                     \
{                                                                           \
    __device__ T* getPointer() {                                            \
        extern __shared__ T ptr_##T##_[];                                   \
        return ptr_##T##_;                                                  \
    }                                                                       \
};

DECLARE_SPECIALIZATIONS(float)
DECLARE_SPECIALIZATIONS(cfloat)
DECLARE_SPECIALIZATIONS(double)
DECLARE_SPECIALIZATIONS(cdouble)
DECLARE_SPECIALIZATIONS(char)
DECLARE_SPECIALIZATIONS(int)
DECLARE_SPECIALIZATIONS(uint)
DECLARE_SPECIALIZATIONS(uchar)

}
