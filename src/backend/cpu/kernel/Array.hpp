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
#include <platform.hpp>

namespace cpu
{
namespace kernel
{

template<typename T>
void evalArray(Array<T> in)
{
    in.setId(cpu::getActiveDeviceId());
    T *ptr = in.data.get();

    af::dim4 odims = in.dims();
    af::dim4 ostrs = in.strides();

    bool is_linear = in.node->isLinear(odims.get());

    if (is_linear) {
        int num = in.elements();
        for (int i = 0; i < num; i++) {
            ptr[i] = *(T *)in.node->calc(i);
        }
    } else {
        for (int w = 0; w < (int)odims[3]; w++) {
            dim_t offw = w * ostrs[3];

            for (int z = 0; z < (int)odims[2]; z++) {
                dim_t offz = z * ostrs[2] + offw;

                for (int y = 0; y < (int)odims[1]; y++) {
                    dim_t offy = y * ostrs[1] + offz;

                    for (int x = 0; x < (int)odims[0]; x++) {
                        dim_t id = x + offy;

                        ptr[id] = *(T *)in.node->calc(x, y, z, w);
                    }
                }
            }
        }
    }
}

}
}
