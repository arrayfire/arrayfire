/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <transform.hpp>
#include <kernel/transform.hpp>
#include <stdexcept>

namespace cuda
{
    template<typename T>
    Array<T>* transform(const Array<T> &in, const Array<float> &transform, const af::dim4 &odims,
                        const bool inverse)
    {
        const af::dim4 idims = in.dims();

        Array<T> *out = createEmptyArray<T>(odims);

        kernel::transform<T>(*out, in, transform, inverse);

        return out;
    }


#define INSTANTIATE(T)                                                                          \
    template Array<T>* transform(const Array<T> &in, const Array<float> &transform,             \
                                 const af::dim4 &odims, const bool inverse);                    \


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}
