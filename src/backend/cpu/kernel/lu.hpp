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

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void lu_split(Param<T> lower, Param<T> upper, CParam<T> in) {
    T *l       = lower.get();
    T *u       = upper.get();
    const T *i = in.get();

    af::dim4 ldm = lower.dims();
    af::dim4 udm = upper.dims();
    af::dim4 idm = in.dims();
    af::dim4 lst = lower.strides();
    af::dim4 ust = upper.strides();
    af::dim4 ist = in.strides();

    for (dim_t ow = 0; ow < idm[3]; ow++) {
        const dim_t lW = ow * lst[3];
        const dim_t uW = ow * ust[3];
        const dim_t iW = ow * ist[3];

        for (dim_t oz = 0; oz < idm[2]; oz++) {
            const dim_t lZW = lW + oz * lst[2];
            const dim_t uZW = uW + oz * ust[2];
            const dim_t iZW = iW + oz * ist[2];

            for (dim_t oy = 0; oy < idm[1]; oy++) {
                const dim_t lYZW = lZW + oy * lst[1];
                const dim_t uYZW = uZW + oy * ust[1];
                const dim_t iYZW = iZW + oy * ist[1];

                for (dim_t ox = 0; ox < idm[0]; ox++) {
                    const dim_t lMem = lYZW + ox;
                    const dim_t uMem = uYZW + ox;
                    const dim_t iMem = iYZW + ox;
                    if (ox > oy) {
                        if (oy < ldm[1]) l[lMem] = i[iMem];
                        if (ox < udm[0]) u[uMem] = scalar<T>(0);
                    } else if (oy > ox) {
                        if (oy < ldm[1]) l[lMem] = scalar<T>(0);
                        if (ox < udm[0]) u[uMem] = i[iMem];
                    } else if (ox == oy) {
                        if (oy < ldm[1]) l[lMem] = scalar<T>(1.0);
                        if (ox < udm[0]) u[uMem] = i[iMem];
                    }
                }
            }
        }
    }
}

void convertPivot(Param<int> p, Param<int> pivot) {
    int *d_pi = pivot.get();
    int *d_po = p.get();
    dim_t d0  = pivot.dims(0);
    for (int j = 0; j < (int)d0; j++) {
        // 1 indexed in pivot
        std::swap(d_po[j], d_po[d_pi[j] - 1]);
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
