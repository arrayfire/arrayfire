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
#include <kernel/range.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    Array<T> range(const dim4& dim, const int seq_dim)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }

        // Set dimension along which the sequence should be
        // Other dimensions are simply tiled
        int _seq_dim = seq_dim;
        if(seq_dim < 0) {
            _seq_dim = 0;   // column wise sequence
        }

        if(_seq_dim < 0 || _seq_dim > 3)
            AF_ERROR("Invalid rep selection", AF_ERR_INVALID_ARG);

        Array<T> out = createEmptyArray<T>(dim);
        kernel::range<T>(out, _seq_dim);

        return out;
    }

#define INSTANTIATE(T)                                                      \
    template Array<T> range<T>(const af::dim4 &dims, const int seq_dims);   \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}
