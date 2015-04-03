/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <identity.hpp>
#include <debug_cuda.hpp>
#include <kernel/identity.hpp>

namespace cuda
{
    template<typename T>
    Array<T> identity(const dim4& dims)
    {
        Array<T> out  = createEmptyArray<T>(dims);
        kernel::identity<T>(out);
        return out;
    }

#define INSTANTIATE_IDENTITY(T)                              \
    template Array<T>  identity<T>    (const af::dim4 &dims);

    INSTANTIATE_IDENTITY(float)
    INSTANTIATE_IDENTITY(double)
    INSTANTIATE_IDENTITY(cfloat)
    INSTANTIATE_IDENTITY(cdouble)
    INSTANTIATE_IDENTITY(int)
    INSTANTIATE_IDENTITY(uint)
    INSTANTIATE_IDENTITY(intl)
    INSTANTIATE_IDENTITY(uintl)
    INSTANTIATE_IDENTITY(char)
    INSTANTIATE_IDENTITY(uchar)

}
