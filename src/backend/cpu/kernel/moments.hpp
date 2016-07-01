/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <Array.hpp>
#include <utility.hpp>
#include <math.hpp>

namespace cpu
{
namespace kernel
{


template<typename T>
void moments(Array<float> &output, Array<T> const &input, af_moment_type moment)
{
    T const * const in       = input.get();
    af::dim4  const idims    = input.dims();

    af::dim4  const ostrides = output.strides();

    float *out = output.get();

    dim_t mId = 0;
    for(dim_t w = 0; w < idims[3]; w++) {
        for(dim_t z = 0; z < idims[2]; z++) {
            dim_t out_off = w * ostrides[3] + z * ostrides[2];
            for(dim_t y = 0; y < idims[1]; y++) {
                for(dim_t x = 0; x < idims[0]; x++) {
                    dim_t m_off=0;
                    float val = in[mId];
                    if((moment & AF_MOMENT_M00) > 0) {
                        out[out_off + m_off] += val;
                        m_off++;
                    }
                    if((moment & AF_MOMENT_M01) > 0) {
                        out[out_off + m_off] += x * val;
                        m_off++;
                    }
                    if((moment & AF_MOMENT_M10) > 0) {
                        out[out_off + m_off] += y * val;
                        m_off++;
                    }
                    if((moment & AF_MOMENT_M11) > 0) {
                        out[out_off + m_off] += x * y * val;
                        m_off++;
                    }
                    mId++;
                }
            }
        }
    }
}


}
}
