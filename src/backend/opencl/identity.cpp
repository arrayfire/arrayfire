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
#include <debug_opencl.hpp>
#include <kernel/identity.hpp>

namespace opencl
{
    template<typename T>
    Array<T> *identity(const dim4& dims)
    {
        Array<T>* out  = createEmptyArray<T>(dims);
        kernel::identity<T>(*out);
        return out;
    }

#define INSTANTIATE_UNIFORM(T)                              \
    template Array<T>*  identity<T>    (const af::dim4 &dims);

    INSTANTIATE_UNIFORM(float)
    INSTANTIATE_UNIFORM(double)
    INSTANTIATE_UNIFORM(cfloat)
    INSTANTIATE_UNIFORM(cdouble)
    INSTANTIATE_UNIFORM(int)
    INSTANTIATE_UNIFORM(uint)
    INSTANTIATE_UNIFORM(char)
    INSTANTIATE_UNIFORM(uchar)

}
