/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <kernel/sort_by_key.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <range.hpp>
#include <reorder.hpp>
#include <sort_by_key.hpp>

namespace arrayfire {
namespace cpu {

template<typename Tk, typename Tv>
void sort_by_key(Array<Tk> &okey, Array<Tv> &oval, const Array<Tk> &ikey,
                 const Array<Tv> &ival, const uint dim, bool isAscending) {
    okey = copyArray<Tk>(ikey);
    oval = copyArray<Tv>(ival);

    switch (dim) {
        case 0:
            getQueue().enqueue(kernel::sort0ByKey<Tk, Tv>, okey, oval,
                               isAscending);
            break;
        case 1:
        case 2:
        case 3:
            getQueue().enqueue(kernel::sortByKeyBatched<Tk, Tv>, okey, oval,
                               dim, isAscending);
            break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    }

    if (dim != 0) {
        af::dim4 preorderDims = okey.dims();
        af::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0]  = okey.dims()[dim];
        for (int i = 1; i <= static_cast<int>(dim); i++) {
            reorderDims[i - 1] = i;
            preorderDims[i]    = okey.dims()[i - 1];
        }

        okey.setDataDims(preorderDims);
        oval.setDataDims(preorderDims);

        okey = reorder<Tk>(okey, reorderDims);
        oval = reorder<Tv>(oval, reorderDims);
    }
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
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, uchar)   \
    INSTANTIATE(Tk, short)   \
    INSTANTIATE(Tk, ushort)  \
    INSTANTIATE(Tk, intl)    \
    INSTANTIATE(Tk, uintl)

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

}  // namespace cpu
}  // namespace arrayfire
