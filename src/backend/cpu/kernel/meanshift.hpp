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
#include <utility.hpp>
#include <type_traits>
#include <vector>

namespace arrayfire {
namespace cpu {
namespace kernel {
template<typename T, bool IsColor>
void meanShift(Param<T> out, CParam<T> in, const float spatialSigma,
               const float chromaticSigma, const unsigned numIterations) {
    typedef typename std::conditional<std::is_same<T, double>::value, double,
                                      float>::type AccType;

    const af::dim4 dims     = in.dims();
    const af::dim4 istrides = in.strides();
    const af::dim4 ostrides = out.strides();
    const unsigned bCount   = (IsColor ? 1 : dims[2]);
    const unsigned channels = (IsColor ? dims[2] : 1);
    const dim_t radius      = std::max((int)(spatialSigma * 1.5f), 1);
    const AccType cvar      = chromaticSigma * chromaticSigma;

    std::array<AccType, 4> currentCenterColors{{0}};
    std::array<AccType, 4> currentMeanColors{{0}};
    std::array<AccType, 4> tempColors{{0}};
    for (dim_t b3 = 0; b3 < dims[3]; ++b3) {
        for (unsigned b2 = 0; b2 < bCount; ++b2) {
            T* outData      = out.get() + b2 * ostrides[2] + b3 * ostrides[3];
            const T* inData = in.get() + b2 * istrides[2] + b3 * istrides[3];

            for (dim_t j = 0; j < dims[1]; ++j) {
                dim_t j_in_off  = j * istrides[1];
                dim_t j_out_off = j * ostrides[1];

                for (dim_t i = 0; i < dims[0]; ++i) {
                    dim_t i_in_off  = i * istrides[0];
                    dim_t i_out_off = i * ostrides[0];

                    for (unsigned ch = 0; ch < channels; ++ch)
                        currentCenterColors[ch] = static_cast<AccType>(
                            inData[j_in_off + i_in_off + ch * istrides[2]]);

                    int meanPosJ = j;
                    int meanPosI = i;

                    // scope of meanshift iterations begin
                    for (unsigned it = 0; it < numIterations; ++it) {
                        int oldMeanPosJ = meanPosJ;
                        int oldMeanPosI = meanPosI;
                        unsigned count  = 0;
                        int shift_y     = 0;
                        int shift_x     = 0;

                        currentMeanColors.fill(0);
                        // Windowing operation
                        for (dim_t wj = -radius; wj <= radius; ++wj) {
                            int hit_count = 0;
                            dim_t tj      = meanPosJ + wj;
                            if (tj < 0 || tj > dims[1] - 1) continue;

                            dim_t tjstride = tj * istrides[1];

                            for (dim_t wi = -radius; wi <= radius; ++wi) {
                                dim_t ti = meanPosI + wi;
                                if (ti < 0 || ti > dims[0] - 1) continue;

                                dim_t tistride = ti * istrides[0];

                                AccType norm = 0;
                                for (unsigned ch = 0; ch < channels; ++ch) {
                                    tempColors[ch] = static_cast<AccType>(
                                        inData[tistride + tjstride +
                                               ch * istrides[2]]);
                                    AccType diff = currentCenterColors[ch] -
                                                   tempColors[ch];
                                    norm += (diff * diff);
                                }
                                if (norm <= cvar) {
                                    for (unsigned ch = 0; ch < channels; ++ch)
                                        currentMeanColors[ch] += tempColors[ch];

                                    shift_x += ti;
                                    ++hit_count;
                                }
                            }
                            count += hit_count;
                            shift_y += tj * hit_count;
                        }

                        if (count == 0) break;

                        const AccType fcount = 1 / static_cast<AccType>(count);

                        meanPosJ =
                            static_cast<int>(std::trunc(shift_y * fcount));
                        meanPosI =
                            static_cast<int>(std::trunc(shift_x * fcount));

                        for (unsigned ch = 0; ch < channels; ++ch)
                            currentMeanColors[ch] =
                                std::trunc(currentMeanColors[ch] * fcount);

                        AccType norm = 0;
                        for (unsigned ch = 0; ch < channels; ++ch) {
                            AccType diff =
                                currentMeanColors[ch] - currentCenterColors[ch];
                            norm += (diff * diff);
                        }

                        // stop the process if mean converged or within given
                        // tolerance range
                        bool stop = (meanPosJ == oldMeanPosJ &&
                                     oldMeanPosI == meanPosI) ||
                                    ((abs(oldMeanPosJ - meanPosJ) +
                                      abs(oldMeanPosI - meanPosI) + norm) <= 1);

                        for (unsigned ch = 0; ch < channels; ++ch)
                            currentCenterColors[ch] = currentMeanColors[ch];

                        if (stop) break;
                    }  // scope of meanshift iterations end

                    for (dim_t ch = 0; ch < channels; ++ch)
                        outData[j_out_off + i_out_off + ch * ostrides[2]] =
                            static_cast<T>(currentCenterColors[ch]);
                }
            }
        }
    }
}
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
