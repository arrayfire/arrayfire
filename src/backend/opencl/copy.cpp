#include <iostream>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>
#include <kernel/memcopy.hpp>
#include <err_opencl.hpp>

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

    template<typename inType, typename outType>
    void copy(Array<outType> &dst, const Array<inType> &src, double factor)
    {
        ARG_ASSERT(1, (src.dims().ndims() == dst.dims().ndims()));

        kernel::copy<inType, outType>(dst, src, src.dims().ndims(), factor);
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

#define INSTANTIATE_COPY(SRC_T)                                                       \
    template void copy<SRC_T, float  >(Array<float  > &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, double >(Array<double > &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, cfloat >(Array<cfloat > &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, cdouble>(Array<cdouble> &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, int    >(Array<int    > &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, uint   >(Array<uint   > &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, uchar  >(Array<uchar  > &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, char   >(Array<char   > &dst, const Array<SRC_T> &src, double factor);

    INSTANTIATE_COPY(float )
    INSTANTIATE_COPY(double)
    INSTANTIATE_COPY(int   )
    INSTANTIATE_COPY(uint  )
    INSTANTIATE_COPY(uchar )
    INSTANTIATE_COPY(char  )

#define INSTANTIATE_COMPLEX_COPY(SRC_T)                                               \
    template void copy<SRC_T, cfloat >(Array<cfloat > &dst, const Array<SRC_T> &src, double factor); \
    template void copy<SRC_T, cdouble>(Array<cdouble> &dst, const Array<SRC_T> &src, double factor);

    INSTANTIATE_COMPLEX_COPY(cfloat )
    INSTANTIATE_COMPLEX_COPY(cdouble)

}
