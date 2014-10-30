/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <sort.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename T, bool DIR>
    void sort(Array<T> &val, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: kernel::sort0<T, DIR>(val, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                                          \
    template void sort<T, true>(Array<T> &val, const Array<T> &in, const unsigned dim);         \
    template void sort<T,false>(Array<T> &val, const Array<T> &in, const unsigned dim);         \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
