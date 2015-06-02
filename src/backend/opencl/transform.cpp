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
#include <Array.hpp>
#include <transform.hpp>
#include <kernel/transform.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename T>
    Array<T> transform(const Array<T> &in, const Array<float> &transform,
                       const af::dim4 &odims,
                       const af_interp_type method, const bool inverse)
    {
        Array<T> out = createEmptyArray<T>(odims);

        if(inverse) {
            switch(method) {
                case AF_INTERP_NEAREST:
                    kernel::transform<T, true, AF_INTERP_NEAREST>
                                     (out, in, transform);
                    break;
                case AF_INTERP_BILINEAR:
                    kernel::transform<T, true, AF_INTERP_BILINEAR>
                                     (out, in, transform);
                    break;
                default:
                    AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
                    break;
            }
        } else {
            switch(method) {
                case AF_INTERP_NEAREST:
                    kernel::transform<T, false, AF_INTERP_NEAREST>
                                     (out, in, transform);
                    break;
                case AF_INTERP_BILINEAR:
                    kernel::transform<T, false, AF_INTERP_BILINEAR>
                                     (out, in, transform);
                    break;
                default:
                    AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
                    break;
            }
        }

        return out;
    }


#define INSTANTIATE(T)                                                  \
    template Array<T> transform(const Array<T> &in, const Array<float> &transform, \
                                const af::dim4 &odims, const af_interp_type method, \
                                const bool inverse);                    \


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
