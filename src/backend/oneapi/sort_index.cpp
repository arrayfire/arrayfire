/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <range.hpp>
#include <reorder.hpp>
#include <sort_index.hpp>
#include <stdexcept>

namespace arrayfire {
namespace oneapi {
template<typename T>
void sort_index(Array<T> &okey, Array<uint> &oval, const Array<T> &in,
                const uint dim, bool isAscending) {
    try {
        // okey contains values, oval contains indices
        okey = copyArray<T>(in);
        oval = range<uint>(in.dims(), dim);
        oval.eval();

        switch (dim) {
            case 0: kernel::sort0ByKey<T, uint>(okey, oval, isAscending); break;
            case 1:
            case 2:
            case 3:
                kernel::sortByKeyBatched<T, uint>(okey, oval, dim, isAscending);
                break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }

        if (dim != 0) {
            af::dim4 preorderDims = okey.dims();
            af::dim4 reorderDims(0, 1, 2, 3);
            reorderDims[dim] = 0;
            preorderDims[0]  = okey.dims()[dim];
            for (uint i = 1; i <= dim; i++) {
                reorderDims[i - 1] = i;
                preorderDims[i]    = okey.dims()[i - 1];
            }

            okey.setDataDims(preorderDims);
            oval.setDataDims(preorderDims);

            okey = reorder<T>(okey, reorderDims);
            oval = reorder<uint>(oval, reorderDims);
        }
    } catch (const std::exception &ex) { AF_ERROR(ex.what(), AF_ERR_INTERNAL); }
}

#define INSTANTIATE(T)                                              \
    template void sort_index<T>(Array<T> & val, Array<uint> & idx,  \
                                const Array<T> &in, const uint dim, \
                                bool isAscending);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(arrayfire::common::half)

}  // namespace oneapi
}  // namespace arrayfire
