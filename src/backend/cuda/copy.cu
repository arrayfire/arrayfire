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
#include <math.hpp>

namespace cuda
{

    template<typename T>
    void copyData(T *data, const Array<T> &A)
    {
        // FIXME: Merge this with copyArray
        evalArray(A);

        Array<T> out = A;
        const T *ptr = NULL;

        if (A.isOwner() || // No offsets, No strides
            A.ndims() == 1 // Simple offset, no strides.
            ) {

            //A.get() gets data with offsets
            ptr = A.get();
        } else {
            //FIXME: Think about implementing eval
            out = copyArray(A);
            ptr = out.get();
        }

        CUDA_CHECK(cudaMemcpy(data, ptr,
                              A.elements() * sizeof(T),
                              cudaMemcpyDeviceToHost));

        return;
    }

    template<typename T>
    Array<T> copyArray(const Array<T> &A)
    {
        Array<T> out = createEmptyArray<T>(A.dims());

        if (A.isLinear()) {
            CUDA_CHECK(cudaMemcpyAsync(out.get(), A.get(),
                                       A.elements() * sizeof(T),
                                       cudaMemcpyDeviceToDevice));
        } else {
            // FIXME: Seems to fail when using Param<T>
            kernel::memcopy(out.get(), out.strides().get(), A.get(), A.dims().get(),
                            A.strides().get(), (uint)A.ndims());
        }
        return out;
    }

    template<typename inType, typename outType>
    Array<outType> padArray(Array<inType> const &in, dim4 const &dims, outType default_value, double factor)
    {
        ARG_ASSERT(1, (in.ndims() == dims.ndims()));
        Array<outType> ret = createEmptyArray<outType>(dims);
        kernel::copy<inType, outType>(ret, in, in.ndims(), default_value, factor);
        return ret;
    }

    template<typename inType, typename outType>
    struct copyWrapper {
        void operator()(Array<outType> &out, Array<inType> const &in)
        {
            kernel::copy<inType, outType>(out, in, in.ndims(), scalar<outType>(0), 1);
        }
    };

    template<typename T>
    struct copyWrapper<T, T> {
        void operator()(Array<T> &out, Array<T> const &in)
        {
            if (out.isLinear() &&
                in.isLinear() &&
                out.elements() == in.elements())
            {
                CUDA_CHECK(cudaMemcpyAsync(out.get(), in.get(),
                                           in.elements() * sizeof(T),
                                           cudaMemcpyDeviceToDevice));
            } else {
                kernel::copy<T, T>(out, in, in.ndims(), scalar<T>(0), 1);
            }
        }
    };

    template<typename inType, typename outType>
    void copyArray(Array<outType> &out, Array<inType> const &in)
    {
        ARG_ASSERT(1, (in.ndims() == out.dims().ndims()));
        copyWrapper<inType, outType> copyFn;
        copyFn(out, in);
    }

#define INSTANTIATE(T)                                              \
    template void      copyData<T> (T *data, const Array<T> &from); \
    template Array<T> copyArray<T>(const Array<T> &A);              \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
    INSTANTIATE(intl   )
    INSTANTIATE(uintl  )

#define INSTANTIATE_PAD_ARRAY(SRC_T)                                    \
    template Array<float  > padArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value, double factor); \
    template Array<double > padArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value, double factor); \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template Array<int    > padArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value, double factor); \
    template Array<uint   > padArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value, double factor); \
    template Array<intl    > padArray<SRC_T, intl    >(Array<SRC_T> const &src, dim4 const &dims, intl     default_value, double factor); \
    template Array<uintl   > padArray<SRC_T, uintl   >(Array<SRC_T> const &src, dim4 const &dims, uintl    default_value, double factor); \
    template Array<uchar  > padArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value, double factor); \
    template Array<char   > padArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value, double factor); \
    template void copyArray<SRC_T, float  >(Array<float  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, double >(Array<double > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cfloat >(Array<cfloat > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, int    >(Array<int    > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uint   >(Array<uint   > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, intl    >(Array<intl    > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uintl   >(Array<uintl   > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uchar  >(Array<uchar  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, char   >(Array<char   > &dst, Array<SRC_T> const &src);

    INSTANTIATE_PAD_ARRAY(float )
    INSTANTIATE_PAD_ARRAY(double)
    INSTANTIATE_PAD_ARRAY(int   )
    INSTANTIATE_PAD_ARRAY(uint  )
    INSTANTIATE_PAD_ARRAY(intl   )
    INSTANTIATE_PAD_ARRAY(uintl  )
    INSTANTIATE_PAD_ARRAY(uchar )
    INSTANTIATE_PAD_ARRAY(char  )

#define INSTANTIATE_PAD_ARRAY_COMPLEX(SRC_T)                            \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template void copyArray<SRC_T, cfloat  >(Array<cfloat  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble   >(Array<cdouble > &dst, Array<SRC_T> const &src);

    INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat )
    INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)

#define SPECILIAZE_UNUSED_COPYARRAY(SRC_T, DST_T) \
    template<> void copyArray<SRC_T, DST_T>(Array<DST_T> &out, Array<SRC_T> const &in) \
    {\
        CUDA_NOT_SUPPORTED();\
    }

    SPECILIAZE_UNUSED_COPYARRAY(cfloat, double)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, float)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, uchar)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, char)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, uint)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, int)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, intl)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, uintl)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, double)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, float)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, uchar)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, char)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, uint)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, int)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, intl)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, uintl)
}
