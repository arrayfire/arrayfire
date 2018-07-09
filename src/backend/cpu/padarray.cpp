/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <type_traits>
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
void multiply_inplace(Array<T> &in, double val)
{
    in.eval();
    getQueue().enqueue(kernel::copyElemwise<T, T>, in, in, 0, val);
}

template<typename inType, typename outType>
Array<outType> padArray(const Array<inType>& in, const dim4& dims,
                        outType default_value, double factor)
{
    Array<outType> ret = createValueArray<outType>(dims, default_value);
    ret.eval();
    in.eval();
    getQueue().enqueue(kernel::copyElemwise<outType, inType>, ret, in, outType(default_value), factor);

    return ret;
}

#define INSTANTIATE(T)                                                  \
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
    template Array<float  > padArray<SRC_T, float  >(const Array<SRC_T>& src, const dim4& dims, float   default_value, double factor); \
    template Array<double > padArray<SRC_T, double >(const Array<SRC_T>& src, const dim4& dims, double  default_value, double factor); \
    template Array<cfloat > padArray<SRC_T, cfloat >(const Array<SRC_T>& src, const dim4& dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(const Array<SRC_T>& src, const dim4& dims, cdouble default_value, double factor); \
    template Array<int    > padArray<SRC_T, int    >(const Array<SRC_T>& src, const dim4& dims, int     default_value, double factor); \
    template Array<uint   > padArray<SRC_T, uint   >(const Array<SRC_T>& src, const dim4& dims, uint    default_value, double factor); \
    template Array<intl   > padArray<SRC_T, intl   >(const Array<SRC_T>& src, const dim4& dims, intl    default_value, double factor); \
    template Array<uintl  > padArray<SRC_T, uintl  >(const Array<SRC_T>& src, const dim4& dims, uintl   default_value, double factor); \
    template Array<short  > padArray<SRC_T, short  >(const Array<SRC_T>& src, const dim4& dims, short   default_value, double factor); \
    template Array<ushort > padArray<SRC_T, ushort >(const Array<SRC_T>& src, const dim4& dims, ushort  default_value, double factor); \
    template Array<uchar  > padArray<SRC_T, uchar  >(const Array<SRC_T>& src, const dim4& dims, uchar   default_value, double factor); \
    template Array<char   > padArray<SRC_T, char   >(const Array<SRC_T>& src, const dim4& dims, char    default_value, double factor); \

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
    template Array<cfloat > padArray<SRC_T, cfloat >(const Array<SRC_T>& src, const dim4& dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(const Array<SRC_T>& src, const dim4& dims, cdouble default_value, double factor); \

INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat )
INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)
}
