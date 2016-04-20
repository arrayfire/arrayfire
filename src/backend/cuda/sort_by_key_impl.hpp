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
#include <reorder.hpp>
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
            case 0: kernel::sort0ByKey<Tk, Tv, isAscending>(okey, oval); break;
            case 1: kernel::sortByKeyBatched<Tk, Tv, isAscending, 1>(okey, oval); break;
            case 2: kernel::sortByKeyBatched<Tk, Tv, isAscending, 2>(okey, oval); break;
            case 3: kernel::sortByKeyBatched<Tk, Tv, isAscending, 3>(okey, oval); break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }

        if(dim != 0) {
            af::dim4 preorderDims = okey.dims();
            af::dim4 reorderDims(0, 1, 2, 3);
            reorderDims[dim] = 0;
            preorderDims[0] = okey.dims()[dim];
            for(int i = 1; i <= (int)dim; i++) {
                reorderDims[i - 1] = i;
                preorderDims[i] = okey.dims()[i - 1];
            }

            okey.setDataDims(preorderDims);
            oval.setDataDims(preorderDims);

            okey = reorder<Tk>(okey, reorderDims);
            oval = reorder<Tv>(oval, reorderDims);
        }
    }

#define INSTANTIATE(Tk, Tv, dr)                                         \
    template void                                                       \
    sort_by_key<Tk, Tv, dr>(Array<Tk> &okey, Array<Tv> &oval,           \
                            const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim); \

#define INSTANTIATE1(Tk    , dr) \
    INSTANTIATE(Tk, float  , dr) \
    INSTANTIATE(Tk, double , dr) \
    INSTANTIATE(Tk, cfloat , dr) \
    INSTANTIATE(Tk, cdouble, dr) \
    INSTANTIATE(Tk, int    , dr) \
    INSTANTIATE(Tk, uint   , dr) \
    INSTANTIATE(Tk, short  , dr) \
    INSTANTIATE(Tk, ushort , dr) \
    INSTANTIATE(Tk, char   , dr) \
    INSTANTIATE(Tk, uchar  , dr) \
    INSTANTIATE(Tk, intl   , dr) \
    INSTANTIATE(Tk, uintl  , dr)
}
