/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <sort_index.hpp>
#include <math.hpp>
#include <algorithm>
#include <numeric>
#include <platform.hpp>
#include <queue.hpp>
#include <range.hpp>
#include <copy.hpp>
#include <reorder.hpp>
#include <kernel/sort_index.hpp>

namespace cpu
{

template<typename T, bool isAscending>
void sort_index(Array<T> &val, Array<uint> &idx, const Array<T> &in, const uint dim)
{
    in.eval();

    val = copyArray<T>(in);
    idx = range<uint>(in.dims(), dim);
    idx.eval();

    switch(dim) {
        case 0: getQueue().enqueue(kernel::sort0Index<T, isAscending>, val, idx); break;
        case 1: getQueue().enqueue(kernel::sortIndexBatched<T, isAscending, 1>, val, idx); break;
        case 2: getQueue().enqueue(kernel::sortIndexBatched<T, isAscending, 2>, val, idx); break;
        case 3: getQueue().enqueue(kernel::sortIndexBatched<T, isAscending, 3>, val, idx); break;
        default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    }

    if(dim != 0) {
        af::dim4 preorderDims = val.dims();
        af::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0] = val.dims()[dim];
        for(int i = 1; i <= (int)dim; i++) {
            reorderDims[i - 1] = i;
            preorderDims[i] = val.dims()[i - 1];
        }

        val.setDataDims(preorderDims);
        idx.setDataDims(preorderDims);

        val = reorder<T>(val, reorderDims);
        idx = reorder<uint>(idx, reorderDims);
    }
}

#define INSTANTIATE(T)                                                  \
    template void sort_index<T, true>(Array<T> &val, Array<uint> &idx, const Array<T> &in, \
                                      const uint dim);                  \
    template void sort_index<T,false>(Array<T> &val, Array<uint> &idx, const Array<T> &in, \
                                      const uint dim);                  \

INSTANTIATE(float)
INSTANTIATE(double)
//INSTANTIATE(cfloat)
//INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)

}
