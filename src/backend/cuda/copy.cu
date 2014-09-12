#include <cuda_runtime_api.h>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>
#include <kernel/memcopy.hpp>
#include <err_cuda.hpp>

namespace cuda
{

    template<typename T>
    void copyData(T *data, const Array<T> &A)
    {
        Array<T> *out = NULL;
        const T *ptr = NULL;

        if (A.isOwner() || // No offsets, No strides
            A.ndims() == 1 // Simple offset, no strides.
            ) {

            //A.get() gets data with offsets
            ptr = A.get();
        } else {
            //FIXME: Think about implementing eval
            out = copyArray(A);
            ptr = out->get();
        }

        CUDA_CHECK(cudaMemcpy(data, ptr,
                              A.elements() * sizeof(T),
                              cudaMemcpyDeviceToHost));

        if (out != NULL) delete out;

        return;
    }


    template<typename T>
    Array<T> *copyArray(const Array<T> &A)
    {
        Array<T> *out = createEmptyArray<T>(A.dims());

        if (A.isOwner()) {
            CUDA_CHECK(cudaMemcpy(out->get(), A.get(),
                                  A.elements() * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
        } else {
            // FIXME: Seems to fail when using Param<T>
            kernel::memcopy(out->get(), out->strides().get(), A.get(), A.dims().get(),
                            A.strides().get(), (uint)A.ndims());
        }
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
