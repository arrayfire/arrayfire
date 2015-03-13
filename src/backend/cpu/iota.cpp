/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <iota.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <algorithm>
#include <numeric>

#include <iostream>
using namespace std;

namespace cpu
{
    ///////////////////////////////////////////////////////////////////////////
    // Kernel Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    void iota(T *out, const dim4 &dims, const dim4 &strides, const dim4 &sdims, const dim4 &tdims)
    {
        for(dim_type w = 0; w < dims[3]; w++) {
            dim_type offW = w * strides[3];
            T valW = (w / tdims[3]) * sdims[2] * sdims[1] * sdims[0];
            for(dim_type z = 0; z < dims[2]; z++) {
                dim_type offWZ = offW + z * strides[2];
                T valZ = valW + (z / tdims[2]) * sdims[1] * sdims[0];
                for(dim_type y = 0; y < dims[1]; y++) {
                    dim_type offWZY = offWZ + y * strides[1];
                    T valY = valZ + (y / tdims[1]) * sdims[0];
                    for(dim_type x = 0; x < dims[0]; x++) {
                        dim_type id = offWZY + x;
                        out[id] = valY + (x % sdims[0]);
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    Array<T> iota(const dim4 &dims, const dim4 &tile_dims)
    {
        dim4 outdims = dims * tile_dims;

        Array<T> out = createEmptyArray<T>(outdims);
        iota<T>(out.get(), out.dims(), out.strides(), dims, tile_dims);

        return out;
    }

#define INSTANTIATE(T)                                                          \
    template Array<T> iota<T>(const af::dim4 &dims, const af::dim4 &tile_dims); \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}
