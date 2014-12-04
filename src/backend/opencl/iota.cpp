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
#include <kernel/iota.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    Array<T> *iota(const dim4& dim, const unsigned rep)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }
        Array<T> *out = createEmptyArray<T>(dim);
        switch(rep) {
            case 0: kernel::iota<T, 0>(*out); break;
            case 1: kernel::iota<T, 1>(*out); break;
            case 2: kernel::iota<T, 2>(*out); break;
            case 3: kernel::iota<T, 3>(*out); break;
            default: AF_ERROR("Invalid rep selection", AF_ERR_INVALID_ARG);
        }
        return out;
    }

#define INSTANTIATE(T)                                                         \
    template Array<T>* iota<T>(const af::dim4 &dims, const unsigned rep);      \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}
