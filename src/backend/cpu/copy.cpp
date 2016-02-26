/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <type_traits>
#include <af/array.h>
#include <Array.hpp>
#include <copy.hpp>
#include <cstring>
#include <algorithm>
#include <complex>
#include <vector>
#include <cassert>
#include <err_cpu.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/copy.hpp>

namespace cpu
{

template<typename T>
void copyData(T *to, const Array<T> &from)
{
    from.eval();
    getQueue().sync();
    if(from.isLinear()) {
        // FIXME: Check for errors / exceptions
        memcpy(to, from.get(), from.elements()*sizeof(T));
    } else {
        dim4 ostrides = calcStrides(from.dims());
        kernel::stridedCopy<T>(to, ostrides, from.get(), from.dims(), from.strides(), from.ndims() - 1);
    }
}

template<typename T>
Array<T> copyArray(const Array<T> &A)
{
    A.eval();
    Array<T> out = createEmptyArray<T>(A.dims());
    getQueue().enqueue(kernel::copy<T, T>, out, A, scalar<T>(0), 1.0);
    return out;
}

template<typename T>
void multiply_inplace(Array<T> &in, double val)
{
    in.eval();
    getQueue().enqueue(kernel::copy<T, T>, in, in, 0, val);
}

template<typename inType, typename outType>
Array<outType> padArray(Array<inType> const &in, dim4 const &dims,
                        outType default_value, double factor)
{
    Array<outType> ret = createValueArray<outType>(dims, default_value);
    ret.eval();
    in.eval();
    getQueue().enqueue(kernel::copy<outType, inType>, ret, in, outType(default_value), factor);
    return ret;
}

template<typename inType, typename outType>
void copyArray(Array<outType> &out, Array<inType> const &in)
{
    out.eval();
    in.eval();
    getQueue().enqueue(kernel::copy<outType, inType>, out, in, scalar<outType>(0), 1.0);
}

#define INSTANTIATE(T)                                                  \
    template void      copyData<T> (T *data, const Array<T> &from);     \
    template Array<T>  copyArray<T>(const Array<T> &A);                 \
    template void      multiply_inplace<T> (Array<T> &in, double norm); \

INSTANTIATE(float  )
INSTANTIATE(double )
INSTANTIATE(cfloat )
INSTANTIATE(cdouble)
INSTANTIATE(int    )
INSTANTIATE(uint   )
INSTANTIATE(uchar  )
INSTANTIATE(char   )
INSTANTIATE(intl   )
INSTANTIATE(uintl  )
INSTANTIATE(short  )
INSTANTIATE(ushort )


#define INSTANTIATE_PAD_ARRAY(SRC_T)                                    \
    template Array<float  > padArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value, double factor); \
    template Array<double > padArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value, double factor); \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template Array<int    > padArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value, double factor); \
    template Array<uint   > padArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value, double factor); \
    template Array<intl   > padArray<SRC_T, intl   >(Array<SRC_T> const &src, dim4 const &dims, intl    default_value, double factor); \
    template Array<uintl  > padArray<SRC_T, uintl  >(Array<SRC_T> const &src, dim4 const &dims, uintl   default_value, double factor); \
    template Array<short  > padArray<SRC_T, short  >(Array<SRC_T> const &src, dim4 const &dims, short   default_value, double factor); \
    template Array<ushort > padArray<SRC_T, ushort >(Array<SRC_T> const &src, dim4 const &dims, ushort  default_value, double factor); \
    template Array<uchar  > padArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value, double factor); \
    template Array<char   > padArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value, double factor); \
    template void copyArray<SRC_T, float  >(Array<float  > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, double >(Array<double > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cfloat >(Array<cfloat > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, int    >(Array<int    > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, uint   >(Array<uint   > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, intl   >(Array<intl   > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, uintl  >(Array<uintl  > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, short  >(Array<short  > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, ushort >(Array<ushort > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, uchar  >(Array<uchar  > &dst, Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, char   >(Array<char   > &dst, Array<SRC_T> const &src);

INSTANTIATE_PAD_ARRAY(float )
INSTANTIATE_PAD_ARRAY(double)
INSTANTIATE_PAD_ARRAY(int   )
INSTANTIATE_PAD_ARRAY(uint  )
INSTANTIATE_PAD_ARRAY(intl  )
INSTANTIATE_PAD_ARRAY(uintl )
INSTANTIATE_PAD_ARRAY(uchar )
INSTANTIATE_PAD_ARRAY(char  )
INSTANTIATE_PAD_ARRAY(ushort)
INSTANTIATE_PAD_ARRAY(short )

#define INSTANTIATE_PAD_ARRAY_COMPLEX(SRC_T)                            \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template void copyArray<SRC_T, cfloat  >(Array<cfloat  > &dst, Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, cdouble   >(Array<cdouble > &dst, Array<SRC_T> const &src);

INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat )
INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)

#define SPECILIAZE_UNUSED_COPYARRAY(SRC_T, DST_T) \
    template<> void copyArray<SRC_T, DST_T>(Array<DST_T> &out, Array<SRC_T> const &in) \
    {\
        CPU_NOT_SUPPORTED();\
    }

SPECILIAZE_UNUSED_COPYARRAY(cfloat , double)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , float)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , uchar)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , char)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , uint)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , int)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , intl)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , uintl)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , short)
SPECILIAZE_UNUSED_COPYARRAY(cfloat , ushort)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, double)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, float)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, uchar)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, char)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, uint)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, int)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, intl)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, uintl)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, short)
SPECILIAZE_UNUSED_COPYARRAY(cdouble, ushort)

}
