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
#include <reorder.hpp>
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
        } catch(std::exception &ex) {
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
    INSTANTIATE(Tk, float  , isAscending)       \
    INSTANTIATE(Tk, double , isAscending)       \
    INSTANTIATE(Tk, cfloat , isAscending)       \
    INSTANTIATE(Tk, cdouble, isAscending)       \
    INSTANTIATE(Tk, int    , isAscending)       \
    INSTANTIATE(Tk, uint   , isAscending)       \
    INSTANTIATE(Tk, char   , isAscending)       \
    INSTANTIATE(Tk, uchar  , isAscending)       \
    INSTANTIATE(Tk, short  , isAscending)       \
    INSTANTIATE(Tk, ushort , isAscending)       \
    INSTANTIATE(Tk, intl   , isAscending)       \
    INSTANTIATE(Tk, uintl  , isAscending)       \

}
