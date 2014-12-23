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
#include <copy.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T, bool isAscending>
    Array<T>* sort(const Array<T> &in, const unsigned dim)
    {
        try {
            Array<T> *out = copyArray<T>(in);
            switch(dim) {
            case 0: kernel::sort0<T, isAscending>(*out);
                break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
            }
            return out;
        } catch (std::exception &ex) {
            AF_ERROR(ex.what(), AF_ERR_INTERNAL);
        }
    }

#define INSTANTIATE(T)                                                  \
    template Array<T>* sort<T, true>(const Array<T> &in, const unsigned dim); \
    template Array<T>*  sort<T,false>(const Array<T> &in, const unsigned dim); \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)

}
