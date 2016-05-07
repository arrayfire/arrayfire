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
#include <copy.hpp>
#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_opencl.hpp>
#include <reorder.hpp>
#include <range.hpp>

namespace opencl
{
    template<typename T, bool isAscending>
    void sort_index(Array<T> &okey, Array<uint> &oval, const Array<T> &in, const uint dim)
    {
        try {
            // okey contains values, oval contains indices
            okey = copyArray<T>(in);
            oval = range<uint>(in.dims(), dim);
            oval.eval();

            switch(dim) {
                case 0: kernel::sort0ByKey<T, uint, isAscending>(okey, oval); break;
                case 1: kernel::sortByKeyBatched<T, uint, isAscending>(okey, oval, 1); break;
                case 2: kernel::sortByKeyBatched<T, uint, isAscending>(okey, oval, 2); break;
                case 3: kernel::sortByKeyBatched<T, uint, isAscending>(okey, oval, 3); break;
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

                okey = reorder<T>(okey, reorderDims);
                oval = reorder<uint>(oval, reorderDims);
            }
        } catch (std::exception &ex) {
            AF_ERROR(ex.what(), AF_ERR_INTERNAL);
        }
    }

#define INSTANTIATE(T)                                                  \
    template void sort_index<T, true>(Array<T> &val, Array<uint> &idx, const Array<T> &in, \
                                      const uint dim);                  \
    template void sort_index<T,false>(Array<T> &val, Array<uint> &idx, const Array<T> &in, \
                                      const uint dim);                  \

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

}
