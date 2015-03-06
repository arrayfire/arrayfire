/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <range.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <algorithm>
#include <numeric>

namespace cpu
{
    ///////////////////////////////////////////////////////////////////////////
    // Kernel Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T, unsigned rep>
    void range(T *out, const dim4 &dim, const dim4 &strides)
    {
        unsigned mul1 = rep > 0;
        unsigned mul2 = rep > 1;
        unsigned mul3 = rep > 2;
        for(dim_type w = 0; w < dim[3]; w++) {
            dim_type offW = w * strides[3];
            for(dim_type z = 0; z < dim[2]; z++) {
                dim_type offWZ = offW + z * strides[2];
                for(dim_type y = 0; y < dim[1]; y++) {
                    dim_type offOffset = offWZ + y * strides[1];

                    T *ptr = out + offOffset;
                    std::iota(ptr, ptr + dim[0],
                             (mul3 * w * dim[0] * dim[1] * dim[2]) +
                             (mul2 * z * dim[0] * dim[1]) +
                             (mul1 * y * dim[0])
                             );
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    Array<T> range(const dim4& dim, const int rep)
    {
        // Repeat highest dimension, ie. creates a single sequence from
        // 0...elements - 1
        int rep_ = rep;
        if(rep < 0) {
            rep_ = dim.ndims() - 1; // ndims = [1,4] => rep = [0, 3]
        }

        Array<T> out = createEmptyArray<T>(dim);
        switch(rep_) {
            case 0: range<T, 0>(out.get(), out.dims(), out.strides()); break;
            case 1: range<T, 1>(out.get(), out.dims(), out.strides()); break;
            case 2: range<T, 2>(out.get(), out.dims(), out.strides()); break;
            case 3: range<T, 3>(out.get(), out.dims(), out.strides()); break;
            default: AF_ERROR("Invalid rep selection", AF_ERR_INVALID_ARG);
        }

        return out;
    }

#define INSTANTIATE(T)                                                  \
    template Array<T> range<T>(const af::dim4 &dims, const int rep);     \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}
