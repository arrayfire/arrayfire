/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
        // FIXME: Merge this with copyArray
        A.eval();

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

    template<typename inType, typename outType>
    void copy(Array<outType> &dst, const Array<inType> &src, outType default_value, double factor)
    {
        ARG_ASSERT(1, (src.dims().ndims() == dst.dims().ndims()));

        kernel::copy(Param<outType>(dst), CParam<inType>(src), src.dims().ndims(), default_value, factor);
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
    template void copy<SRC_T, float  >(Array<float  > &dst, const Array<SRC_T> &src, float   default_value, double factor); \
    template void copy<SRC_T, double >(Array<double > &dst, const Array<SRC_T> &src, double  default_value, double factor); \
    template void copy<SRC_T, cfloat >(Array<cfloat > &dst, const Array<SRC_T> &src, cfloat  default_value, double factor); \
    template void copy<SRC_T, cdouble>(Array<cdouble> &dst, const Array<SRC_T> &src, cdouble default_value, double factor); \
    template void copy<SRC_T, int    >(Array<int    > &dst, const Array<SRC_T> &src, int     default_value, double factor); \
    template void copy<SRC_T, uint   >(Array<uint   > &dst, const Array<SRC_T> &src, uint    default_value, double factor); \
    template void copy<SRC_T, uchar  >(Array<uchar  > &dst, const Array<SRC_T> &src, uchar   default_value, double factor); \
    template void copy<SRC_T, char   >(Array<char   > &dst, const Array<SRC_T> &src, char    default_value, double factor);

    INSTANTIATE_COPY(float )
    INSTANTIATE_COPY(double)
    INSTANTIATE_COPY(int   )
    INSTANTIATE_COPY(uint  )
    INSTANTIATE_COPY(uchar )
    INSTANTIATE_COPY(char  )

#define INSTANTIATE_COMPLEX_COPY(SRC_T)                                               \
    template void copy<SRC_T, cfloat >(Array<cfloat > &dst, const Array<SRC_T> &src, cfloat  default_value, double factor); \
    template void copy<SRC_T, cdouble>(Array<cdouble> &dst, const Array<SRC_T> &src, cdouble default_value, double factor);

    INSTANTIATE_COMPLEX_COPY(cfloat )
    INSTANTIATE_COMPLEX_COPY(cdouble)

#define INSTANTIATE_UNSUPPORTED_COMPLEX_COPY(cmplxType, T)              \
    template<> void copy(Array<T> &dst, const Array<cfloat> &src,       \
                                    T  default_value, double factor)    \
    {                                                                   \
        TYPE_ERROR(0,(af_dtype) af::dtype_traits<T>::af_type);          \
    }                                                                   \
    template<> void copy(Array<T> &dst, const Array<cdouble> &src,      \
                                        T default_value, double factor) \
    {                                                                   \
        TYPE_ERROR(0,(af_dtype) af::dtype_traits<T>::af_type);          \
    }                                                                   \

    INSTANTIATE_UNSUPPORTED_COMPLEX_COPY(cfloat, double)
    INSTANTIATE_UNSUPPORTED_COMPLEX_COPY(cfloat, float)
    INSTANTIATE_UNSUPPORTED_COMPLEX_COPY(cfloat, int)
    INSTANTIATE_UNSUPPORTED_COMPLEX_COPY(cfloat, uint)
    INSTANTIATE_UNSUPPORTED_COMPLEX_COPY(cfloat, char)
    INSTANTIATE_UNSUPPORTED_COMPLEX_COPY(cfloat, uchar)

}
