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
#include <err_opencl.hpp>
#include <where.hpp>
#include <complex>
#include <kernel/where.hpp>

namespace opencl
{
    template<typename T>
    Array<uint> where(const Array<T> &in)
    {
        Param Out;
        Param In = in;
        kernel::where<T>(Out, In);
        return createParamArray<uint>(Out);
    }


#define INSTANTIATE(T)                                  \
    template Array<uint> where<T>(const Array<T> &in);  \

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
