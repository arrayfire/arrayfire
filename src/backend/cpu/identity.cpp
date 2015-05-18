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
#include <math.hpp>

namespace cpu
{
    template<typename T>
    Array<T> identity(const dim4& dims)
    {
        Array<T> out = createEmptyArray<T>(dims);
        T *ptr = out.get();
        const dim_t *out_dims  = out.dims().get();

        for (dim_t k = 0; k < out_dims[2] * out_dims[3]; k++) {
            for (dim_t j = 0; j < out_dims[1]; j++) {
                for (dim_t i = 0; i < out_dims[0]; i++) {
                    ptr[j * out_dims[0] + i]  = (i == j) ? scalar<T>(1) : scalar<T>(0);
                }
            }
            ptr += out_dims[0] * out_dims[1];
        }
        return out;
    }

#define INSTANTIATE_IDENTITY(T)                              \
    template Array<T>  identity<T>    (const af::dim4 &dims);

    INSTANTIATE_IDENTITY(float)
    INSTANTIATE_IDENTITY(double)
    INSTANTIATE_IDENTITY(cfloat)
    INSTANTIATE_IDENTITY(cdouble)
    INSTANTIATE_IDENTITY(int)
    INSTANTIATE_IDENTITY(intl)
    INSTANTIATE_IDENTITY(uintl)
    INSTANTIATE_IDENTITY(uint)
    INSTANTIATE_IDENTITY(char)
    INSTANTIATE_IDENTITY(uchar)

}
