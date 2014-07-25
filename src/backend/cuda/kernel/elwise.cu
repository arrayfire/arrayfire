#include <functional>
#include <af/defines.h>
#include <kernel/elwise.hpp>

namespace cuda
{
namespace kernel
{

template<typename T>
__global__
void setKernel(T* ptr, T val, const size_t elements)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < elements) {
        ptr[idx] = val;
    }
}

template<typename T>
void set(T* ptr, T val, const size_t &elements)
{
    dim3 threads(512);
    dim3 blocks(divup(elements,threads.x));
    typedef typename af::dtype_traits<T>::cuda_type     cuType;
    typedef typename af::dtype_traits<T>::cuda_type_ptr cuTypePtr;

    setKernel<cuType><<<blocks, threads>>>(reinterpret_cast<cuTypePtr>(ptr), *reinterpret_cast<cuTypePtr>(&val), elements);
}

template void set<float>(float* ptr, float val, const size_t &elements);
template void set<double>(double* ptr, double val, const size_t &elements);
template void set<cfloat>(cfloat* ptr, cfloat val, const size_t &elements);
template void set<cdouble>(cdouble* ptr, cdouble val, const size_t &elements);
template void set<char>(char* ptr, char val, const size_t &elements);
template void set<int>(int* ptr, int val, const size_t &elements);
template void set<unsigned>(unsigned* ptr, unsigned val, const size_t &elements);
template void set<unsigned char>(unsigned char* ptr, unsigned char val, const size_t &elements);

}
}
