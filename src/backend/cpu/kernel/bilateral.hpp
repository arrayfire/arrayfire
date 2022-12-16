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
#include <utility.hpp>
#include <cmath>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename OutT, typename InT>
void bilateral(Param<OutT> out, CParam<InT> in, float const s_sigma,
               float const c_sigma) {
    af::dim4 const dims     = in.dims();
    af::dim4 const istrides = in.strides();
    af::dim4 const ostrides = out.strides();

    // clamp spatical and chromatic sigma's
    float space_       = std::min(11.5f, std::max(s_sigma, 0.f));
    float color_       = std::max(c_sigma, 0.f);
    dim_t const radius = std::max((dim_t)(space_ * 1.5f), (dim_t)1);
    float const svar   = space_ * space_;
    float const cvar   = color_ * color_;

    for (dim_t b3 = 0; b3 < dims[3]; ++b3) {
        OutT *outData     = out.get() + b3 * ostrides[3];
        InT const *inData = in.get() + b3 * istrides[3];

        // b3 for loop handles following batch configurations
        //  - gfor
        //  - input based batch
        //      - when input is 4d array for color images
        for (dim_t b2 = 0; b2 < dims[2]; ++b2) {
            // b2 for loop handles following batch configurations
            //  - channels
            //  - input based batch
            //      - when input is 3d array for grayscale images
            for (dim_t j = 0; j < dims[1]; ++j) {
                // j steps along 2nd dimension
                for (dim_t i = 0; i < dims[0]; ++i) {
                    // i steps along 1st dimension
                    OutT norm         = 0.0;
                    OutT res          = 0.0;
                    OutT const center = (OutT)inData[getIdx(istrides, i, j)];
                    for (dim_t wj = -radius; wj <= radius; ++wj) {
                        // clamps offsets
                        dim_t tj = clamp(j + wj, dim_t(0), dims[1] - 1);
                        for (dim_t wi = -radius; wi <= radius; ++wi) {
                            // clamps offsets
                            dim_t ti = clamp(i + wi, dim_t(0), dims[0] - 1);
                            // proceed
                            OutT const val =
                                (OutT)inData[getIdx(istrides, ti, tj)];
                            OutT const gauss_space =
                                (wi * wi + wj * wj) / (-2.0 * svar);
                            OutT const gauss_range =
                                ((center - val) * (center - val)) /
                                (-2.0 * cvar);
                            OutT const weight =
                                std::exp(gauss_space + gauss_range);
                            norm += weight;
                            res += val * weight;
                        }
                    }  // filter loop ends here

                    outData[getIdx(ostrides, i, j)] = res / norm;
                }  // 1st dimension loop ends here
            }      // 2nd dimension loop ends here
            outData += ostrides[2];
            inData += istrides[2];
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
