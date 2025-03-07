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
#include <math.hpp>

#include <cassert>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename Ti, typename To, bool isDX>
void derivative(Param<To> output, CParam<Ti> input) {
    const af::dim4 dims     = input.dims();
    const af::dim4 istrides = input.strides();
    const af::dim4 ostrides = output.strides();

    auto reflect101 = [](int index, int endIndex) -> int {
        return std::abs(endIndex - std::abs(endIndex - index));
    };

    for (dim_t b3 = 0; b3 < dims[3]; ++b3) {
        To* optr       = output.get() + b3 * ostrides[3];
        const Ti* iptr = input.get() + b3 * istrides[3];
        for (dim_t b2 = 0; b2 < dims[2]; ++b2) {
            for (dim_t j = 0; j < dims[1]; ++j) {
                int joff    = j;
                int _joff   = reflect101(j - 1, static_cast<int>(dims[1] - 1));
                int joff_   = reflect101(j + 1, static_cast<int>(dims[1] - 1));
                int joffset = j * ostrides[1];

                for (dim_t i = 0; i < dims[0]; ++i) {
                    To accum = To(0);

                    int ioff = i;
                    int _ioff =
                        reflect101(i - 1, static_cast<int>(dims[0] - 1));
                    int ioff_ =
                        reflect101(i + 1, static_cast<int>(dims[0] - 1));

                    To NW = iptr[_joff * istrides[1] + _ioff * istrides[0]];
                    To SW = iptr[_joff * istrides[1] + ioff_ * istrides[0]];
                    To NE = iptr[joff_ * istrides[1] + _ioff * istrides[0]];
                    To SE = iptr[joff_ * istrides[1] + ioff_ * istrides[0]];

                    if (isDX) {
                        To N  = iptr[joff * istrides[1] + _ioff * istrides[0]];
                        To S  = iptr[joff * istrides[1] + ioff_ * istrides[0]];
                        accum = SW + SE - (NW + NE) + 2 * (S - N);
                    } else {
                        To W  = iptr[_joff * istrides[1] + ioff * istrides[0]];
                        To E  = iptr[joff_ * istrides[1] + ioff * istrides[0]];
                        accum = NE + SE - (NW + SW) + 2 * (E - W);
                    }

                    optr[joffset + i * ostrides[0]] = accum;
                }
            }

            optr += ostrides[2];
            iptr += istrides[2];
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
