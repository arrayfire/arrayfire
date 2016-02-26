/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <Array.hpp>
#include <math.hpp>
#include <algorithm>
#include <numeric>
#include <queue>
#include <err_cpu.hpp>
#include <functional>

namespace cpu
{
namespace kernel
{

template<typename Tk, typename Tv, bool isAscending>
void sort0_by_key(Array<Tk> okey, Array<Tv> oval, Array<uint> oidx,
                  const Array<Tk> ikey, const Array<Tv> ival)
{
    function<bool(Tk, Tk)> op = std::greater<Tk>();
    if(isAscending) { op = std::less<Tk>(); }

    // Get pointers and initialize original index locations
        uint *oidx_ptr = oidx.get();
          Tk *okey_ptr = okey.get();
          Tv *oval_ptr = oval.get();
    const Tk *ikey_ptr = ikey.get();
    const Tv *ival_ptr = ival.get();

    std::vector<uint> seq_vec(oidx.dims()[0]);
    std::iota(seq_vec.begin(), seq_vec.end(), 0);

    const Tk *comp_ptr = nullptr;
    auto comparator = [&comp_ptr, &op](size_t i1, size_t i2) {return op(comp_ptr[i1], comp_ptr[i2]);};

    for(dim_t w = 0; w < ikey.dims()[3]; w++) {
        dim_t okeyW = w * okey.strides()[3];
        dim_t ovalW = w * oval.strides()[3];
        dim_t oidxW = w * oidx.strides()[3];
        dim_t ikeyW = w * ikey.strides()[3];
        dim_t ivalW = w * ival.strides()[3];

        for(dim_t z = 0; z < ikey.dims()[2]; z++) {
            dim_t okeyWZ = okeyW + z * okey.strides()[2];
            dim_t ovalWZ = ovalW + z * oval.strides()[2];
            dim_t oidxWZ = oidxW + z * oidx.strides()[2];
            dim_t ikeyWZ = ikeyW + z * ikey.strides()[2];
            dim_t ivalWZ = ivalW + z * ival.strides()[2];

            for(dim_t y = 0; y < ikey.dims()[1]; y++) {

                dim_t okeyOffset = okeyWZ + y * okey.strides()[1];
                dim_t ovalOffset = ovalWZ + y * oval.strides()[1];
                dim_t oidxOffset = oidxWZ + y * oidx.strides()[1];
                dim_t ikeyOffset = ikeyWZ + y * ikey.strides()[1];
                dim_t ivalOffset = ivalWZ + y * ival.strides()[1];

                uint *ptr = oidx_ptr + oidxOffset;
                std::copy(seq_vec.begin(), seq_vec.end(), ptr);

                comp_ptr = ikey_ptr + ikeyOffset;
                std::stable_sort(ptr, ptr + ikey.dims()[0], comparator);

                for (dim_t i = 0; i < oval.dims()[0]; ++i){
                    uint sortIdx = oidx_ptr[oidxOffset + i];
                    okey_ptr[okeyOffset + i] = ikey_ptr[ikeyOffset + sortIdx];
                    oval_ptr[ovalOffset + i] = ival_ptr[ivalOffset + sortIdx];
                }
            }
        }
    }

    return;
}

}
}
