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
#include <iota.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <range.hpp>
#include <reorder.hpp>
#include <sort.hpp>
#include <sort_by_key.hpp>
#include <algorithm>
#include <functional>

namespace arrayfire {
namespace cpu {

template<typename T, int dim>
void sortBatched(Array<T>& val, bool isAscending) {
    af::dim4 inDims = val.dims();

    // Sort dimension
    af::dim4 tileDims(1);
    af::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    Array<uint> key = iota<uint>(seqDims, tileDims);

    Array<uint> resKey = createEmptyArray<uint>(dim4());
    Array<T> resVal    = createEmptyArray<T>(dim4());

    val.setDataDims(inDims.elements());
    key.setDataDims(inDims.elements());

    sort_by_key<T, uint>(resVal, resKey, val, key, 0, isAscending);

    // Needs to be ascending (true) in order to maintain the indices properly
    sort_by_key<uint, T>(key, val, resKey, resVal, 0, true);
    val.setDataDims(inDims);  // This is correct only for dim0
}

template<typename T>
void sort0(Array<T>& val, bool isAscending) {
    int higherDims = val.elements() / val.dims()[0];
    // TODO Make a better heurisitic
    if (higherDims > 10) {
        sortBatched<T, 0>(val, isAscending);
    } else {
        getQueue().enqueue(kernel::sort0Iterative<T>, val, isAscending);
    }
}

template<typename T>
Array<T> sort(const Array<T>& in, const unsigned dim, bool isAscending) {
    Array<T> out = copyArray<T>(in);
    switch (dim) {
        case 0: sort0<T>(out, isAscending); break;
        case 1: sortBatched<T, 1>(out, isAscending); break;
        case 2: sortBatched<T, 2>(out, isAscending); break;
        case 3: sortBatched<T, 3>(out, isAscending); break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    }

    if (dim != 0) {
        af::dim4 preorderDims = out.dims();
        af::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0]  = out.dims()[dim];
        for (int i = 1; i <= static_cast<int>(dim); i++) {
            reorderDims[i - 1] = i;
            preorderDims[i]    = out.dims()[i - 1];
        }

        out.setDataDims(preorderDims);
        out = reorder<T>(out, reorderDims);
    }
    return out;
}

#define INSTANTIATE(T)                                                \
    template Array<T> sort<T>(const Array<T>& in, const unsigned dim, \
                              bool isAscending);

INSTANTIATE(float)
INSTANTIATE(double)
// INSTANTIATE(cfloat)
// INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)

}  // namespace cpu
}  // namespace arrayfire
