#include <iostream>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>
#include <kernel/memcopy.hpp>

namespace opencl
{

    template<typename T>
    void copyData(T *data, const Array<T> &A)
    {
        dim_type offset = 0;
        cl::Buffer buf;
        Array<T> *out = nullptr;

        if (A.isOwner() || // No offsets, No strides
            A.ndims() == 1 // Simple offset, no strides.
            ) {
            buf = A.get();
            offset = A.getOffset();
        } else {
            //FIXME: Think about implementing eval
            out = copyArray(A);
            buf = out->get();
            offset = 0;
        }

        //FIXME: Add checks
        getQueue().enqueueReadBuffer(buf, CL_TRUE,
                                      sizeof(T) * offset,
                                      sizeof(T) * A.elements(),
                                      data);
        if (out != nullptr) delete out;

        return;
    }

    template<typename T>
    Array<T> *copyArray(const Array<T> &A)
    {
        Array<T> *out = createEmptyArray<T>(A.dims());
        dim_type offset = A.getOffset();

        if (A.isOwner()) {
            // FIXME: Add checks
            getQueue().enqueueCopyBuffer(A.get(), out->get(),
                                          sizeof(T) * offset, 0,
                                          A.elements() * sizeof(T));
        } else {
            kernel::memcopy<T>(out->get(), out->strides().get(), A.get(), A.dims().get(),
                               A.strides().get(), offset, (uint)A.ndims());
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
