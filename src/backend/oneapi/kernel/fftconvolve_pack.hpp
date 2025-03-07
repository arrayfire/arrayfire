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

#include <iostream>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename inputType, typename outputType>
class fftconvolve_packCreateKernel {
   public:
    fftconvolve_packCreateKernel(write_accessor<outputType> d_out, KParam oInfo,
                                 read_accessor<inputType> d_in, KParam iInfo,
                                 const int di0_half, const int odd_di0)
        : d_out_(d_out)
        , oInfo_(oInfo)
        , d_in_(d_in)
        , iInfo_(iInfo)
        , di0_half_(di0_half)
        , odd_di0_(odd_di0) {}
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

        const int to0 = t % so1;
        const int to1 = (t / so1) % do1;
        const int to2 = (t / so2) % do2;
        const int to3 = t / so3;

        // const int di0 = iInfo_.dims[0];
        const int di1 = iInfo_.dims[1];
        const int di2 = iInfo_.dims[2];

        const int si1 = iInfo_.strides[1];
        const int si2 = iInfo_.strides[2];
        const int si3 = iInfo_.strides[3];

        const int ti0 = to0;
        const int ti1 = to1 * si1;
        const int ti2 = to2 * si2;
        const int ti3 = to3 * si3;

        const int iidx1 = iInfo_.offset + ti3 + ti2 + ti1 + ti0;
        const int iidx2 = iidx1 + di0_half_;

        // Treating complex output array as real-only array,
        // thus, multiply strides by 2
        const int oidx1 = oInfo_.offset + to3 * so3 * 2 + to2 * so2 * 2 +
                          to1 * so1 * 2 + to0 * 2;
        const int oidx2 = oidx1 + 1;

        if (to0 < di0_half_ && to1 < di1 && to2 < di2) {
            d_out_[oidx1] = (outputType)d_in_[iidx1];
            if (ti0 == di0_half_ - 1 && odd_di0_ == 1)
                d_out_[oidx2] = (outputType)0;
            else
                d_out_[oidx2] = (outputType)d_in_[iidx2];
        } else {
            // Pad remaining elements with 0s
            d_out_[oidx1] = (outputType)0;
            d_out_[oidx2] = (outputType)0;
        }
    }

   private:
    write_accessor<outputType> d_out_;
    KParam oInfo_;
    read_accessor<inputType> d_in_;
    KParam iInfo_;
    const int di0_half_;
    const int odd_di0_;
};

template<typename convT, typename T>
void packDataHelper(Param<convT> packed, Param<T> sig, Param<T> filter,
                    const int rank, AF_BATCH_KIND kind) {
    Param<T> sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, rank, kind);

    int sig_packed_elem = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];

    // Number of packed complex elements in dimension 0
    int sig_half_d0     = divup(sig.info.dims[0], 2);
    int sig_half_d0_odd = sig.info.dims[0] % 2;

    int blocks = divup(sig_packed_elem, THREADS);

    // Locate features kernel sizes
    auto local  = sycl::range(THREADS);
    auto global = sycl::range(blocks * THREADS);

    // Treat complex output as an array of scalars
    using convScalarT    = typename convT::value_type;
    auto packed_num_elem = (*packed.data).get_range().size();
    auto sig_tmp_buffer  = (*packed.data)
                              .template reinterpret<convScalarT>(
                                  sycl::range<1>{packed_num_elem * 2});

    getQueue().submit([&](auto &h) {
        read_accessor<T> d_sig                = {*sig.data, h};
        write_accessor<convScalarT> d_sig_tmp = {sig_tmp_buffer, h};
        h.parallel_for(sycl::nd_range{global, local},
                       fftconvolve_packCreateKernel<T, convScalarT>(
                           d_sig_tmp, sig_tmp.info, d_sig, sig.info,
                           sig_half_d0, sig_half_d0_odd));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
