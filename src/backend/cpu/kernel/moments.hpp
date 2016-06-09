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


template<typename T, af_moment_type MM>
struct moments_op
{
    T operator()(T const * const in, dim_t mId, dim_t const idx, dim_t const idy)
    {
        return;
    }
};

template<typename T>
struct moments_op<T, AF_MOMENT_M00>
{
    T operator()(T const * const in, dim_t mId, dim_t const idx, dim_t const idy)
    {
        return in[mId];
    }
};

template<typename T>
struct moments_op<T, AF_MOMENT_M01>
{
    T operator()(T const * const in, dim_t mId, dim_t const idx, dim_t const idy)
    {
        return idx * in[mId];
    }
};

template<typename T>
struct moments_op<T, AF_MOMENT_M10>
{
    T operator()(T const * const in, dim_t mId, dim_t const idx, dim_t const idy)
    {
        return idy * in[mId];
    }
};

template<typename T>
struct moments_op<T, AF_MOMENT_M11>
{
    T operator()(T const * const in, dim_t mId, dim_t const idx, dim_t const idy)
    {
        return idx * idy * in[mId];
    }
};

template<typename T, af_moment_type Method>
void moments(Array<float> &output, Array<T> const input)
{
    T const * const in       = input.get();
    af::dim4  const idims    = input.dims();
    af::dim4  const istrides = input.strides();
    dim_t     const iElems   = input.elements();

    af::dim4  const odims    = output.dims();

    moments_op<T, Method> op;
    bool pBatch = !(idims[2] == 1 && idims[3] == 1);
    bool tDim   = (idims[3] != 1);
    bool zDim   = (idims[2] != 1);

    float *out = output.get();

    dim_t mId = 0;
    for(dim_t w = 0; w < idims[3]; w++) {
        for(dim_t z = 0; z < idims[2]; z++) {
            T val = scalar<T>(0);
            for(dim_t y = 0; y < idims[1]; y++) {
                for(dim_t x = 0; x < idims[0]; x++) {
                    val += op(in, mId, x, y);
                    mId++;
                }
            }
            out[w * odims[0] + z] = (float)val;
        }
    }
}


}
}
