/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <sort.hpp>
#include <copy.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <reorder.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    Array<T> sort(const Array<T> &in, const unsigned dim, bool isAscending)
    {
        try {
            Array<T> out = copyArray<T>(in);
            switch(dim) {
                case 0: kernel::sort0<T>(out, isAscending); break;
                case 1: kernel::sortBatched<T, 1>(out, isAscending); break;
                case 2: kernel::sortBatched<T, 2>(out, isAscending); break;
                case 3: kernel::sortBatched<T, 3>(out, isAscending); break;
                default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
            }

            if(dim != 0) {
                af::dim4 preorderDims = out.dims();
                af::dim4 reorderDims(0, 1, 2, 3);
                reorderDims[dim] = 0;
                preorderDims[0] = out.dims()[dim];
                for(int i = 1; i <= (int)dim; i++) {
                    reorderDims[i - 1] = i;
                    preorderDims[i] = out.dims()[i - 1];
                }

                out.setDataDims(preorderDims);
                out = reorder<T>(out, reorderDims);
            }
            return out;
        } catch (std::exception &ex) {
            AF_ERROR(ex.what(), AF_ERR_INTERNAL);
        }
    }

#define INSTANTIATE(T)                                                  \
    template Array<T> sort<T>(const Array<T> &in, const unsigned dim, bool isAscending);

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
