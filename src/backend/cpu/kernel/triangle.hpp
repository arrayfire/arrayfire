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

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T, bool IsUpper, bool IsUnitDiag>
void triangle(Param<T> out, CParam<T> in) {
    T *o       = out.get();
    const T *i = in.get();

    af::dim4 odm = out.dims();

    af::dim4 ost = out.strides();
    af::dim4 ist = in.strides();

    for (dim_t ow = 0; ow < odm[3]; ow++) {
        const dim_t oW = ow * ost[3];
        const dim_t iW = ow * ist[3];

        for (dim_t oz = 0; oz < odm[2]; oz++) {
            const dim_t oZW = oW + oz * ost[2];
            const dim_t iZW = iW + oz * ist[2];

            for (dim_t oy = 0; oy < odm[1]; oy++) {
                const dim_t oYZW = oZW + oy * ost[1];
                const dim_t iYZW = iZW + oy * ist[1];

                for (dim_t ox = 0; ox < odm[0]; ox++) {
                    const dim_t oMem = oYZW + ox;
                    const dim_t iMem = iYZW + ox;

                    bool cond         = IsUpper ? (oy >= ox) : (oy <= ox);
                    bool do_unit_diag = (IsUnitDiag && ox == oy);
                    if (cond) {
                        o[oMem] = do_unit_diag ? scalar<T>(1) : i[iMem];
                    } else {
                        o[oMem] = scalar<T>(0);
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
