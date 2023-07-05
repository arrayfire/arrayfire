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

template<typename inputType, typename outputType>
class fftconvolve_padCreateKernel {
   public:
    fftconvolve_padCreateKernel(write_accessor<outputType> d_out, KParam oInfo,
                                read_accessor<inputType> d_in, KParam iInfo)
        : d_out_(d_out), oInfo_(oInfo), d_in_(d_in), iInfo_(iInfo) {}
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
        const int to3 = (t / so3);

        const int di0 = iInfo_.dims[0];
        const int di1 = iInfo_.dims[1];
        const int di2 = iInfo_.dims[2];
        const int di3 = iInfo_.dims[3];

        const int si1 = iInfo_.strides[1];
        const int si2 = iInfo_.strides[2];
        const int si3 = iInfo_.strides[3];

        const int ti0 = to0;
        const int ti1 = to1 * si1;
        const int ti2 = to2 * si2;
        const int ti3 = to3 * si3;

        const int iidx = iInfo_.offset + ti3 + ti2 + ti1 + ti0;

        const int oidx = oInfo_.offset + t * 2;

        if (to0 < di0 && to1 < di1 && to2 < di2 && to3 < di3) {
            // Copy input elements to real elements, set imaginary elements to 0
            d_out_[oidx]     = (outputType)d_in_[iidx];
            d_out_[oidx + 1] = (outputType)0;
        } else {
            // Pad remaining of the matrix to 0s
            d_out_[oidx]     = (outputType)0;
            d_out_[oidx + 1] = (outputType)0;
        }
    }

   private:
    write_accessor<outputType> d_out_;
    KParam oInfo_;
    read_accessor<inputType> d_in_;
    KParam iInfo_;
};

template<typename convT, typename T>
void padDataHelper(Param<convT> packed, Param<T> sig, Param<T> filter,
                   const int rank, AF_BATCH_KIND kind) {
    Param<T> sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, rank, kind);

    int filter_packed_elem =
        filter_tmp.info.strides[3] * filter_tmp.info.dims[3];

    int blocks = divup(filter_packed_elem, THREADS);

    // Locate features kernel sizes
    auto local  = sycl::range(THREADS);
    auto global = sycl::range(blocks * THREADS);

    // Treat complex output as an array of scalars
    using convScalarT      = typename convT::value_type;
    auto packed_num_elem   = (*packed.data).get_range().size();
    auto filter_tmp_buffer = (*packed.data)
                                 .template reinterpret<convScalarT>(
                                     sycl::range<1>{packed_num_elem * 2});

    getQueue().submit([&](auto &h) {
        read_accessor<T> d_filter = {*filter.data, h, sycl::read_only};
        write_accessor<convScalarT> d_filter_tmp = {filter_tmp_buffer, h};
        h.parallel_for(
            sycl::nd_range{global, local},
            fftconvolve_padCreateKernel<T, convScalarT>(
                d_filter_tmp, filter_tmp.info, d_filter, filter.info));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
