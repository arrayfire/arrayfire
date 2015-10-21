/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <copy.hpp>
#include <sort_by_key.hpp>
#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename Tk, typename Tv, bool isAscending>
    void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
               const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim)
    {
        okey = copyArray<Tk>(ikey);
        oval = copyArray<Tv>(ival);
        switch(dim) {
        case 0: kernel::sort0_by_key<Tk, Tv, isAscending>(okey, oval);
            break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(Tk, Tv, dr)                                         \
    template void                                                       \
    sort_by_key<Tk, Tv, dr>(Array<Tk> &okey, Array<Tv> &oval,           \
                            const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim); \

#define INSTANTIATE1(Tk,    dr) \
    INSTANTIATE(Tk, float,  dr) \
    INSTANTIATE(Tk, double, dr) \
    INSTANTIATE(Tk, int,    dr) \
    INSTANTIATE(Tk, uint,   dr) \
    INSTANTIATE(Tk, short,  dr) \
    INSTANTIATE(Tk, ushort, dr) \
    INSTANTIATE(Tk, char,   dr) \
    INSTANTIATE(Tk, uchar,  dr) \
    INSTANTIATE(Tk, intl,   dr) \
    INSTANTIATE(Tk, uintl,  dr)
}
