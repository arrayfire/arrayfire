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
#include <err_cpu.hpp>
#include <functional>

namespace cpu
{
namespace kernel
{

template<typename T, bool isAscending>
void sort0_index(Array<T> val, Array<uint> idx, const Array<T> in)
{
    // initialize original index locations
       uint *idx_ptr = idx.get();
          T *val_ptr = val.get();
    const T *in_ptr  = in.get();
    function<bool(T, T)> op = std::greater<T>();
    if(isAscending) { op = std::less<T>(); }

    std::vector<uint> seq_vec(idx.dims()[0]);
    std::iota(seq_vec.begin(), seq_vec.end(), 0);

    const T *comp_ptr = nullptr;
    auto comparator = [&comp_ptr, &op](size_t i1, size_t i2) {return op(comp_ptr[i1], comp_ptr[i2]);};

    for(dim_t w = 0; w < in.dims()[3]; w++) {
        dim_t valW = w * val.strides()[3];
        dim_t idxW = w * idx.strides()[3];
        dim_t  inW = w *  in.strides()[3];
        for(dim_t z = 0; z < in.dims()[2]; z++) {
            dim_t valWZ = valW + z * val.strides()[2];
            dim_t idxWZ = idxW + z * idx.strides()[2];
            dim_t  inWZ =  inW + z *  in.strides()[2];
            for(dim_t y = 0; y < in.dims()[1]; y++) {

                dim_t valOffset = valWZ + y * val.strides()[1];
                dim_t idxOffset = idxWZ + y * idx.strides()[1];
                dim_t inOffset  =  inWZ + y *  in.strides()[1];

                uint *ptr = idx_ptr + idxOffset;
                std::copy(seq_vec.begin(), seq_vec.end(), ptr);

                comp_ptr = in_ptr + inOffset;
                std::stable_sort(ptr, ptr + in.dims()[0], comparator);

                for (dim_t i = 0; i < val.dims()[0]; ++i){
                    val_ptr[valOffset + i] = in_ptr[inOffset + idx_ptr[idxOffset + i]];
                }
            }
        }
    }

    return;
}

}
}
