/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <err_cpu.hpp>
#include <math.hpp>
#include <algorithm>
#include <functional>
#include <numeric>

namespace arrayfire {
namespace cpu {
namespace kernel {

// Based off of http://stackoverflow.com/a/12399290
template<typename T>
void sort0Iterative(Param<T> val, bool isAscending) {
    // initialize original index locations
    T *val_ptr = val.get();

    std::function<bool(T, T)> op = std::greater<T>();
    if (isAscending) { op = std::less<T>(); }

    T *comp_ptr = nullptr;
    for (dim_t w = 0; w < val.dims(3); w++) {
        dim_t valW = w * val.strides(3);
        for (dim_t z = 0; z < val.dims(2); z++) {
            dim_t valWZ = valW + z * val.strides(2);
            for (dim_t y = 0; y < val.dims(1); y++) {
                dim_t valOffset = valWZ + y * val.strides(1);

                comp_ptr = val_ptr + valOffset;
                std::sort(comp_ptr, comp_ptr + val.dims(0), op);
            }
        }
    }
    return;
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
