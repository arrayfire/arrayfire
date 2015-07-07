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
    Array<T> transform(const Array<T> &in, const Array<float> &transform, const af::dim4 &odims,
                        const af_interp_type method, const bool inverse)
    {
        const af::dim4 idims = in.dims();

        Array<T> out = createEmptyArray<T>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                kernel::transform<T, AF_INTERP_NEAREST> (out, in, transform, inverse);
                break;
            case AF_INTERP_BILINEAR:
                kernel::transform<T, AF_INTERP_BILINEAR>(out, in, transform, inverse);
                break;
            case AF_INTERP_LOWER:
                kernel::transform<T, AF_INTERP_LOWER>   (out, in, transform, inverse);
                break;
            default:
                AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
        }

        return out;
    }


#define INSTANTIATE(T)                                                                      \
    template Array<T> transform(const Array<T> &in, const Array<float> &transform,          \
                                const af::dim4 &odims, const af_interp_type method,         \
                                const bool inverse);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
