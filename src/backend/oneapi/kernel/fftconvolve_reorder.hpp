/*******************************************************
 * Copyright (c) 2023, ArrayFire
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
#include <af/defines.h>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T, typename convScalarT>
class fftconvolve_reorderCreateKernel {
   public:
    fftconvolve_reorderCreateKernel(write_accessor<T> d_out, KParam oInfo,
                                    read_accessor<convScalarT> d_in,
                                    KParam iInfo, KParam fInfo,
                                    const int half_di0, const int baseDim,
                                    const int fftScale, const bool EXPAND,
                                    const bool ROUND_OUT)
        : d_out_(d_out)
        , oInfo_(oInfo)
        , d_in_(d_in)
        , iInfo_(iInfo)
        , fInfo_(fInfo)
        , half_di0_(half_di0)
        , baseDim_(baseDim)
        , fftScale_(fftScale)
        , EXPAND_(EXPAND)
        , ROUND_OUT_(ROUND_OUT) {}
    void operator()(sycl::nd_item<1> it) const {
        const int t = it.get_global_id(0);

        const int tMax = oInfo_.strides[3] * oInfo_.dims[3];

        if (t >= tMax) return;

        // const int do0 = oInfo_.dims[0];
        const int do1 = oInfo_.dims[1];
        const int do2 = oInfo_.dims[2];

        const int so1 = oInfo_.strides[1];
        const int so2 = oInfo_.strides[2];
        const int so3 = oInfo_.strides[3];

        // Treating complex input array as real-only array,
        // thus, multiply dimension 0 and strides by 2
        const int si1 = iInfo_.strides[1] * 2;
        const int si2 = iInfo_.strides[2] * 2;
        const int si3 = iInfo_.strides[3] * 2;

        const int to0 = t % so1;
        const int to1 = (t / so1) % do1;
        const int to2 = (t / so2) % do2;
        const int to3 = (t / so3);

        int oidx = to3 * so3 + to2 * so2 + to1 * so1 + to0;

        int ti0, ti1, ti2, ti3;
        if (EXPAND_) {
            ti0 = to0;
            ti1 = to1 * si1;
            ti2 = to2 * si2;
            ti3 = to3 * si3;
        } else {
            ti0 = to0 + fInfo_.dims[0] / 2;
            ti1 = (to1 + (baseDim_ > 1) * (fInfo_.dims[1] / 2)) * si1;
            ti2 = (to2 + (baseDim_ > 2) * (fInfo_.dims[2] / 2)) * si2;
            ti3 = to3 * si3;
        }

        // Divide output elements to cuFFT resulting scale, round result if
        // output type is single or double precision floating-point
        if (ti0 < half_di0_) {
            // Copy top elements
            int iidx = iInfo_.offset + ti3 + ti2 + ti1 + ti0 * 2;
            if (ROUND_OUT_)
                d_out_[oidx] = (T)round(d_in_[iidx] / fftScale_);
            else
                d_out_[oidx] = (T)(d_in_[iidx] / fftScale_);
        } else if (ti0 < half_di0_ + fInfo_.dims[0] - 1) {
            // Add central elements
            int iidx1 = iInfo_.offset + ti3 + ti2 + ti1 + ti0 * 2;
            int iidx2 =
                iInfo_.offset + ti3 + ti2 + ti1 + (ti0 - half_di0_) * 2 + 1;
            if (ROUND_OUT_)
                d_out_[oidx] =
                    (T)round((d_in_[iidx1] + d_in_[iidx2]) / fftScale_);
            else
                d_out_[oidx] = (T)((d_in_[iidx1] + d_in_[iidx2]) / fftScale_);
        } else {
            // Copy bottom elements
            const int iidx =
                iInfo_.offset + ti3 + ti2 + ti1 + (ti0 - half_di0_) * 2 + 1;
            if (ROUND_OUT_)
                d_out_[oidx] = (T)round(d_in_[iidx] / fftScale_);
            else
                d_out_[oidx] = (T)(d_in_[iidx] / fftScale_);
        }
    }

   private:
    write_accessor<T> d_out_;
    KParam oInfo_;
    read_accessor<convScalarT> d_in_;
    KParam iInfo_;
    KParam fInfo_;
    const int half_di0_;
    const int baseDim_;
    const int fftScale_;
    const bool EXPAND_;
    const bool ROUND_OUT_;
};

template<typename T, typename convT>
void reorderOutputHelper(Param<T> out, Param<convT> packed, Param<T> sig,
                         Param<T> filter, const int rank, AF_BATCH_KIND kind,
                         bool expand) {
    int fftScale = 1;

    // Calculate the scale by which to divide clFFT results
    for (int k = 0; k < rank; k++) fftScale *= packed.info.dims[k];

    Param<T> sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, rank, kind);

    // Number of packed complex elements in dimension 0
    int sig_half_d0 = divup(sig.info.dims[0], 2);

    int blocks = divup(out.info.strides[3] * out.info.dims[3], THREADS);

    constexpr bool round_out = std::is_integral<T>::value;

    auto local  = sycl::range(THREADS);
    auto global = sycl::range(blocks * THREADS);

    using convScalarT = typename convT::value_type;

    if (kind == AF_BATCH_RHS) {
        auto packed_num_elem   = (*packed.data).get_range().size();
        auto filter_tmp_buffer = (*packed.data)
                                     .template reinterpret<convScalarT>(
                                         sycl::range<1>{packed_num_elem * 2});
        getQueue().submit([&](auto &h) {
            read_accessor<convScalarT> d_filter_tmp = {filter_tmp_buffer, h};
            write_accessor<T> d_out = {*out.data, h, sycl::write_only};
            h.parallel_for(
                sycl::nd_range{global, local},
                fftconvolve_reorderCreateKernel<T, convScalarT>(
                    d_out, out.info, d_filter_tmp, filter_tmp.info, filter.info,
                    sig_half_d0, rank, fftScale, expand, round_out));
        });
    } else {
        auto packed_num_elem = (*packed.data).get_range().size();
        auto sig_tmp_buffer  = (*packed.data)
                                  .template reinterpret<convScalarT>(
                                      sycl::range<1>{packed_num_elem * 2});
        getQueue().submit([&](auto &h) {
            read_accessor<convScalarT> d_sig_tmp = {sig_tmp_buffer, h,
                                                    sycl::read_only};
            write_accessor<T> d_out              = {*out.data, h};
            h.parallel_for(
                sycl::nd_range{global, local},
                fftconvolve_reorderCreateKernel<T, convScalarT>(
                    d_out, out.info, d_sig_tmp, sig_tmp.info, filter.info,
                    sig_half_d0, rank, fftScale, expand, round_out));
        });
    }

    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
