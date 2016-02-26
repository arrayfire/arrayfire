/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <sort_by_key.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/sort_by_key.hpp>

namespace cpu
{

template<typename Tk, typename Tv, bool isAscending>
void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
           const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim)
{
    ikey.eval();
    ival.eval();

    okey = createEmptyArray<Tk>(ikey.dims());
    oval = createEmptyArray<Tv>(ival.dims());
    Array<uint> oidx = createValueArray(ikey.dims(), 0u);
    oidx.eval();

    switch(dim) {
        case 0: getQueue().enqueue(kernel::sort0_by_key<Tk, Tv, isAscending>,
                                   okey, oval, oidx, ikey, ival); break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    }
}

#define INSTANTIATE(Tk, Tv)                                             \
    template void                                                       \
    sort_by_key<Tk, Tv, true>(Array<Tk> &okey, Array<Tv> &oval,         \
                              const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim); \
    template void                                                       \
    sort_by_key<Tk, Tv,false>(Array<Tk> &okey, Array<Tv> &oval,         \
                              const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim); \

#define INSTANTIATE1(Tk)       \
    INSTANTIATE(Tk, float)     \
    INSTANTIATE(Tk, double)    \
    INSTANTIATE(Tk, int)       \
    INSTANTIATE(Tk, uint)      \
    INSTANTIATE(Tk, char)      \
    INSTANTIATE(Tk, uchar)     \
    INSTANTIATE(Tk, short)     \
    INSTANTIATE(Tk, ushort)    \
    INSTANTIATE(Tk, intl)      \
    INSTANTIATE(Tk, uintl)     \


INSTANTIATE1(float)
INSTANTIATE1(double)
INSTANTIATE1(int)
INSTANTIATE1(uint)
INSTANTIATE1(char)
INSTANTIATE1(uchar)
INSTANTIATE1(short)
INSTANTIATE1(ushort)
INSTANTIATE1(intl)
INSTANTIATE1(uintl)

}
