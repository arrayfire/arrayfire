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
    template<typename T, int dim>
    void range(T *out, const dim4 &dims, const dim4 &strides)
    {
        for(dim_t w = 0; w < dims[3]; w++) {
            dim_t offW = w * strides[3];
            for(dim_t z = 0; z < dims[2]; z++) {
                dim_t offWZ = offW + z * strides[2];
                for(dim_t y = 0; y < dims[1]; y++) {
                    dim_t offWZY = offWZ + y * strides[1];
                    for(dim_t x = 0; x < dims[0]; x++) {
                        dim_t id = offWZY + x;
                        if(dim == 0) {
                            out[id] = x;
                        } else if(dim == 1) {
                            out[id] = y;
                        } else if(dim == 2) {
                            out[id] = z;
                        } else if(dim == 3) {
                            out[id] = w;
                        }
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    Array<T> range(const dim4& dims, const int seq_dim)
    {
        // Set dimension along which the sequence should be
        // Other dimensions are simply tiled
        int _seq_dim = seq_dim;
        if(seq_dim < 0) {
            _seq_dim = 0;   // column wise sequence
        }

        Array<T> out = createEmptyArray<T>(dims);
        switch(_seq_dim) {
            case 0: range<T, 0>(out.get(), out.dims(), out.strides()); break;
            case 1: range<T, 1>(out.get(), out.dims(), out.strides()); break;
            case 2: range<T, 2>(out.get(), out.dims(), out.strides()); break;
            case 3: range<T, 3>(out.get(), out.dims(), out.strides()); break;
            default : AF_ERROR("Invalid rep selection", AF_ERR_ARG);
        }


        return out;
    }

#define INSTANTIATE(T)                                                      \
    template Array<T> range<T>(const af::dim4 &dims, const int seq_dims);   \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(uchar)
}
