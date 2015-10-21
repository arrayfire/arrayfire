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
#include <err_opencl.hpp>

namespace opencl
{
    template<typename Tk, typename Tv, bool isAscending>
    void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
               const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim)
    {
        try {
            okey = copyArray<Tk>(ikey);
            oval = copyArray<Tv>(ival);
            switch(dim) {
            case 0: kernel::sort0_by_key<Tk, Tv, isAscending>(okey, oval);
                break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
            }
        }catch(std::exception &ex) {
            AF_ERROR(ex.what(), AF_ERR_INTERNAL);
        }
    }

#define INSTANTIATE(Tk, Tv, isAscending)                                \
    template void                                                       \
    sort_by_key<Tk, Tv, isAscending>(Array<Tk> &okey, Array<Tv> &oval,  \
                                     const Array<Tk> &ikey,             \
                                     const Array<Tv> &ival,             \
                                     const unsigned dim);               \


#define INSTANTIATE1(Tk, isAscending)           \
    INSTANTIATE(Tk, float , isAscending)        \
    INSTANTIATE(Tk, double, isAscending)        \
    INSTANTIATE(Tk, int   , isAscending)        \
    INSTANTIATE(Tk, uint  , isAscending)        \
    INSTANTIATE(Tk, char  , isAscending)        \
    INSTANTIATE(Tk, uchar , isAscending)        \
    INSTANTIATE(Tk, short , isAscending)        \
    INSTANTIATE(Tk, ushort, isAscending)        \
    INSTANTIATE(Tk, intl  , isAscending)        \
    INSTANTIATE(Tk, uintl , isAscending)        \

}
