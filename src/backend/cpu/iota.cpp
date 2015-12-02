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
#include <platform.hpp>
#include <async_queue.hpp>

using namespace std;

namespace cpu
{
///////////////////////////////////////////////////////////////////////////
// Kernel Functions
///////////////////////////////////////////////////////////////////////////
template<typename T>
void iota_(Array<T>  output, const dim4 &sdims, const dim4 &tdims)
{
    const dim4 dims    = output.dims();
    T* out             = output.get();
    const dim4 strides = output.strides();

    for(dim_t w = 0; w < dims[3]; w++) {
        dim_t offW = w * strides[3];
        T valW = (w % sdims[3]) * sdims[0] * sdims[1] * sdims[2];
        for(dim_t z = 0; z < dims[2]; z++) {
            dim_t offWZ = offW + z * strides[2];
            T valZ = valW + (z % sdims[2]) * sdims[0] * sdims[1];
            for(dim_t y = 0; y < dims[1]; y++) {
                dim_t offWZY = offWZ + y * strides[1];
                T valY = valZ + (y % sdims[1]) * sdims[0];
                for(dim_t x = 0; x < dims[0]; x++) {
                    dim_t id = offWZY + x;
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

    getQueue().enqueue(iota_<T>, out, dims, tile_dims);

    return out;
}

#define INSTANTIATE(T)                                                          \
    template Array<T> iota<T>(const af::dim4 &dims, const af::dim4 &tile_dims); \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
}
