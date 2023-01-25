/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
//#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <reorder.hpp>
#include <sort_by_key.hpp>
#include <stdexcept>

namespace arrayfire {
namespace oneapi {
template<typename Tk, typename Tv>
void sort_by_key(Array<Tk> &okey, Array<Tv> &oval, const Array<Tk> &ikey,
                 const Array<Tv> &ival, const unsigned dim, bool isAscending) {
    ONEAPI_NOT_SUPPORTED("");
}

#define INSTANTIATE(Tk, Tv)                                        \
    template void sort_by_key<Tk, Tv>(                             \
        Array<Tk> & okey, Array<Tv> & oval, const Array<Tk> &ikey, \
        const Array<Tv> &ival, const uint dim, bool isAscending);

#define INSTANTIATE1(Tk)     \
    INSTANTIATE(Tk, float)   \
    INSTANTIATE(Tk, double)  \
    INSTANTIATE(Tk, cfloat)  \
    INSTANTIATE(Tk, cdouble) \
    INSTANTIATE(Tk, int)     \
    INSTANTIATE(Tk, uint)    \
    INSTANTIATE(Tk, short)   \
    INSTANTIATE(Tk, ushort)  \
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, uchar)   \
    INSTANTIATE(Tk, intl)    \
    INSTANTIATE(Tk, uintl)

INSTANTIATE1(float)
INSTANTIATE1(double)
INSTANTIATE1(int)
INSTANTIATE1(uint)
INSTANTIATE1(short)
INSTANTIATE1(ushort)
INSTANTIATE1(char)
INSTANTIATE1(uchar)
INSTANTIATE1(intl)
INSTANTIATE1(uintl)
}  // namespace oneapi
}  // namespace arrayfire
