/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>

#undef _GLIBCXX_USE_INT128
#include <where.hpp>
#include <complex>
#include <kernel/where.hpp>

namespace cuda
{
    template<typename T>
    Array<uint> where(const Array<T> &in)
    {
        Param<uint> out;
        kernel::where<T>(out, in);
        return createParamArray<uint>(out);
    }


#define INSTANTIATE(T)                                  \
    template Array<uint> where<T>(const Array<T> &in);    \

    INSTANTIATE(float  )
    INSTANTIATE(cfloat )
    INSTANTIATE(double )
    INSTANTIATE(cdouble)
    INSTANTIATE(char   )
    INSTANTIATE(int    )
    INSTANTIATE(uint   )
    INSTANTIATE(intl   )
    INSTANTIATE(uintl  )
    INSTANTIATE(uchar  )

}
