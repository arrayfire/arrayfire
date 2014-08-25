#include <iostream>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>

namespace opencl
{

    template<typename T>
    void copyData(T *data, const Array<T> &A)
    {
        //FIXME: Add checks
        //FIXME: Check if faster or slower than cl::enqueueReadBuffer
        getQueue(0).enqueueReadBuffer(A.get(),CL_TRUE,0,sizeof(T)*A.elements(),data);
    }

    template<typename T>
    Array<T> *copyArray(const Array<T> &A)
    {
        Array<T> *out = createEmptyArray<T>(A.dims());
        // FIXME: Add checks
        getQueue(0).enqueueCopyBuffer(A.get(), out->get(), 0, 0, A.elements() * sizeof(T));
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
