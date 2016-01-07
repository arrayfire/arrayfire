/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <Array.hpp>
#include <math.hpp>
#include <err_cpu.hpp>

namespace cpu
{
namespace kernel
{

template<typename T, af_interp_type method>
void rotate(Array<T> output, const Array<T> input, const float theta)
{
    const af::dim4 odims    = output.dims();
    const af::dim4 idims    = input.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    const T* in   = input.get();
          T* out  = output.get();
    dim_t nimages = idims[2];

    void (*t_fn)(T *, const T *, const float *, const af::dim4 &,
                 const af::dim4 &, const af::dim4 &,
                 const dim_t, const dim_t, const dim_t, const dim_t,
                 const bool);

    const float c = cos(-theta), s = sin(-theta);
    float tx, ty;
    {
        const float nx = 0.5 * (idims[0] - 1);
        const float ny = 0.5 * (idims[1] - 1);
        const float mx = 0.5 * (odims[0] - 1);
        const float my = 0.5 * (odims[1] - 1);
        const float sx = (mx * c + my *-s);
        const float sy = (mx * s + my * c);
        tx = -(sx - nx);
        ty = -(sy - ny);
    }

    const float tmat[6] = {std::round( c * 1000) / 1000.0f,
                           std::round(-s * 1000) / 1000.0f,
                           std::round(tx * 1000) / 1000.0f,
                           std::round( s * 1000) / 1000.0f,
                           std::round( c * 1000) / 1000.0f,
                           std::round(ty * 1000) / 1000.0f,
                          };

    switch(method) {
        case AF_INTERP_NEAREST:
            t_fn = &transform_n;
            break;
        case AF_INTERP_BILINEAR:
            t_fn = &transform_b;
            break;
        case AF_INTERP_LOWER:
            t_fn = &transform_l;
            break;
        default:
            AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
            break;
    }


    // Do transform for image
    for(int yy = 0; yy < (int)odims[1]; yy++) {
        for(int xx = 0; xx < (int)odims[0]; xx++) {
            t_fn(out, in, tmat, idims, ostrides, istrides, nimages, 0, xx, yy, false);
        }
    }
}

}
}
