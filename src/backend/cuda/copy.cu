#include <cassert>

#include <cuda_runtime_api.h>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>

namespace cuda
{

    template<typename T>
    void copyData(T *data, const Array<T> &A)
    {
        //FIXME: Add checks
        cudaMemcpy(data, A.get(), A.elements()*sizeof(T), cudaMemcpyDeviceToHost);

        return;
    }


    template<typename T>
    Array<T> *copyArray(const Array<T> &A)
    {
        Array<T> *out = createEmptyArray<T>(A.dims());

        // FIXME: Add checks
        cudaMemcpy(out->get(), A.get(), A.elements()*sizeof(T), cudaMemcpyDeviceToDevice);
        return out;
    }


#define INSTANTIATE(T)                                                  \
    template void      copyData<T> (T *data, const Array<T> &from);     \
    template Array<T>* copyArray<T>(const Array<T> &A);                 \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
