/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>
#include <kernel/memcopy.hpp>
#include <err_opencl.hpp>
#include <math.hpp>

namespace opencl
{

    template<typename T>
    void copyData(T *data, const Array<T> &A)
    {

        // FIXME: Merge this with copyArray
        A.eval();

        dim_type offset = 0;
        cl::Buffer buf;
        Array<T> out = A;

        if (A.isOwner() || // No offsets, No strides
            A.ndims() == 1 // Simple offset, no strides.
            ) {
            buf = *A.get();
            offset = A.getOffset();
        } else {
            //FIXME: Think about implementing eval
            out = copyArray(A);
            buf = *out.get();
            offset = 0;
        }

        //FIXME: Add checks
        getQueue().enqueueReadBuffer(buf, CL_TRUE,
                                     sizeof(T) * offset,
                                     sizeof(T) * A.elements(),
                                     data);
        return;
    }

    template<typename T>
    Array<T> copyArray(const Array<T> &A)
    {
        Array<T> out = createEmptyArray<T>(A.dims());
        dim_type offset = A.getOffset();

        if (A.isOwner()) {
            // FIXME: Add checks
            getQueue().enqueueCopyBuffer(*A.get(), *out.get(),
                                         sizeof(T) * offset, 0,
                                         A.elements() * sizeof(T));
        } else {
            kernel::memcopy<T>(*out.get(), out.strides().get(), *A.get(), A.dims().get(),
                               A.strides().get(), offset, (uint)A.ndims());
        }
        return out;
    }

    template<typename inType, typename outType>
    Array<outType> padArray(Array<inType> const &in, dim4 const &dims, outType default_value, double factor)
    {
        Array<outType> ret = createEmptyArray<outType>(dims);

        if (in.dims() == dims)
            kernel::copy<inType, outType, true >(ret, in, in.ndims(), default_value, factor);
        else
            kernel::copy<inType, outType, false>(ret, in, in.ndims(), default_value, factor);
        return ret;
    }

    template<typename inType, typename outType>
    void copyArray(Array<outType> &out, Array<inType> const &in)
    {
        if (in.dims() == out.dims())
            kernel::copy<inType, outType, true >(out, in, in.ndims(), scalar<outType>(0), 1);
        else
            kernel::copy<inType, outType, false>(out, in, in.ndims(), scalar<outType>(0), 1);
    }

#define INSTANTIATE(T)                                              \
    template void      copyData<T> (T *data, const Array<T> &from); \
    template Array<T>  copyArray<T>(const Array<T> &A);             \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)

    #define INSTANTIATE_PAD_ARRAY(SRC_T)                                    \
    template Array<float  > padArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value, double factor); \
    template Array<double > padArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value, double factor); \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template Array<int    > padArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value, double factor); \
    template Array<uint   > padArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value, double factor); \
    template Array<uchar  > padArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value, double factor); \
    template Array<char   > padArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value, double factor); \
    template void copyArray<SRC_T, float  >(Array<float  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, double >(Array<double > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cfloat >(Array<cfloat > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, int    >(Array<int    > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uint   >(Array<uint   > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uchar  >(Array<uchar  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, char   >(Array<char   > &dst, Array<SRC_T> const &src);

    INSTANTIATE_PAD_ARRAY(float )
    INSTANTIATE_PAD_ARRAY(double)
    INSTANTIATE_PAD_ARRAY(int   )
    INSTANTIATE_PAD_ARRAY(uint  )
    INSTANTIATE_PAD_ARRAY(uchar )
    INSTANTIATE_PAD_ARRAY(char  )

#define INSTANTIATE_PAD_ARRAY_COMPLEX(SRC_T)                            \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template void copyArray<SRC_T, cfloat  >(Array<cfloat  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble   >(Array<cdouble > &dst, Array<SRC_T> const &src);

    INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat )
    INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)


}
