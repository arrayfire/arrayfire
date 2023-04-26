/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

inline int convert_int_rtz(float number) { return ((int)(number)); }

template<typename T, typename AccType, const int MAX_CHANNELS>
class meanshiftCreateKernel {
   public:
    meanshiftCreateKernel(write_accessor<T> d_dst, KParam oInfo,
                          read_accessor<T> d_src, KParam iInfo, int radius,
                          float cvar, unsigned numIters, int nBBS0, int nBBS1)
        : d_dst_(d_dst)
        , oInfo_(oInfo)
        , d_src_(d_src)
        , iInfo_(iInfo)
        , radius_(radius)
        , cvar_(cvar)
        , numIters_(numIters)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        unsigned b2 = g.get_group_id(0) / nBBS0_;
        unsigned b3 = g.get_group_id(1) / nBBS1_;
        const int gx =
            g.get_local_range(0) * (g.get_group_id(0) - b2 * nBBS0_) +
            it.get_local_id(0);
        const int gy =
            g.get_local_range(1) * (g.get_group_id(1) - b3 * nBBS1_) +
            it.get_local_id(1);

        if (gx < iInfo_.dims[0] && gy < iInfo_.dims[1]) {
            const T* iptr =
                d_src_.get_pointer() + (b2 * iInfo_.strides[2] +
                                        b3 * iInfo_.strides[3] + iInfo_.offset);
            T* optr = d_dst_.get_pointer() +
                      (b2 * oInfo_.strides[2] + b3 * oInfo_.strides[3]);

            int meanPosI = gx;
            int meanPosJ = gy;

            T currentCenterColors[MAX_CHANNELS];
            T tempColors[MAX_CHANNELS];

            AccType currentMeanColors[MAX_CHANNELS];

#pragma unroll
            for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                currentCenterColors[ch] =
                    iptr[gx * iInfo_.strides[0] + gy * iInfo_.strides[1] +
                         ch * iInfo_.strides[2]];

            const int dim0LenLmt = iInfo_.dims[0] - 1;
            const int dim1LenLmt = iInfo_.dims[1] - 1;

            // scope of meanshift iterationd begin
            for (uint it = 0; it < numIters_; ++it) {
                int oldMeanPosJ = meanPosJ;
                int oldMeanPosI = meanPosI;
                unsigned count  = 0;

                int shift_x = 0;
                int shift_y = 0;

                for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                    currentMeanColors[ch] = 0;

                for (int wj = -radius_; wj <= radius_; ++wj) {
                    int hit_count = 0;
                    int tj        = meanPosJ + wj;

                    if (tj < 0 || tj > dim1LenLmt) continue;

                    for (int wi = -radius_; wi <= radius_; ++wi) {
                        int ti = meanPosI + wi;

                        if (ti < 0 || ti > dim0LenLmt) continue;

                        AccType norm = 0;
#pragma unroll
                        for (int ch = 0; ch < MAX_CHANNELS; ++ch) {
                            unsigned idx = ti * iInfo_.strides[0] +
                                           tj * iInfo_.strides[1] +
                                           ch * iInfo_.strides[2];
                            tempColors[ch] = iptr[idx];
                            AccType diff   = (AccType)currentCenterColors[ch] -
                                           (AccType)tempColors[ch];
                            norm += (diff * diff);
                        }

                        if (norm <= cvar_) {
#pragma unroll
                            for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                                currentMeanColors[ch] +=
                                    (AccType)tempColors[ch];

                            shift_x += ti;
                            ++hit_count;
                        }
                    }
                    count += hit_count;
                    shift_y += tj * hit_count;
                }

                if (count == 0) break;

                const AccType fcount = 1 / (AccType)count;

                meanPosI = convert_int_rtz(shift_x * fcount);
                meanPosJ = convert_int_rtz(shift_y * fcount);

#pragma unroll
                for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                    currentMeanColors[ch] =
                        convert_int_rtz(currentMeanColors[ch] * fcount);

                AccType norm = 0;
#pragma unroll
                for (int ch = 0; ch < MAX_CHANNELS; ++ch) {
                    AccType diff = (AccType)currentCenterColors[ch] -
                                   currentMeanColors[ch];
                    norm += (diff * diff);
                }

                bool stop =
                    (meanPosJ == oldMeanPosJ && meanPosI == oldMeanPosI) ||
                    ((abs(oldMeanPosJ - meanPosJ) +
                      abs(oldMeanPosI - meanPosI)) +
                     norm) <= 1;

#pragma unroll
                for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                    currentCenterColors[ch] = (T)(currentMeanColors[ch]);

                if (stop) break;
            }  // scope of meanshift iterations end

#pragma unroll
            for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                optr[gx * oInfo_.strides[0] + gy * oInfo_.strides[1] +
                     ch * oInfo_.strides[2]] = currentCenterColors[ch];
        }
    }

   private:
    write_accessor<T> d_dst_;
    KParam oInfo_;
    read_accessor<T> d_src_;
    KParam iInfo_;
    int radius_;
    float cvar_;
    unsigned numIters_;
    int nBBS0_;
    int nBBS1_;
};

template<typename T>
void meanshift(Param<T> out, const Param<T> in, const float spatialSigma,
               const float chromaticSigma, const uint numIters,
               const bool is_color) {
    using AccType = typename std::conditional<std::is_same<T, double>::value,
                                              double, float>::type;
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    const int MAX_CHANNELS = (is_color ? 3 : 1);

    auto local = sycl::range(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    const int bCount = (is_color ? 1 : in.info.dims[2]);

    auto global = sycl::range(bCount * blk_x * THREADS_X,
                              in.info.dims[3] * blk_y * THREADS_Y);

    // clamp spatial and chromatic sigma's
    int radius = std::max((int)(spatialSigma * 1.5f), 1);

    const float cvar = chromaticSigma * chromaticSigma;

    getQueue().submit([&](auto& h) {
        read_accessor<T> d_src{*in.data, h};
        write_accessor<T> d_dst{*out.data, h};
        if (MAX_CHANNELS == 3) {
            h.parallel_for(sycl::nd_range{global, local},
                           meanshiftCreateKernel<T, AccType, 3>(
                               d_dst, out.info, d_src, in.info, radius, cvar,
                               numIters, blk_x, blk_y));
        } else {
            h.parallel_for(sycl::nd_range{global, local},
                           meanshiftCreateKernel<T, AccType, 1>(
                               d_dst, out.info, d_src, in.info, radius, cvar,
                               numIters, blk_x, blk_y));
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
