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

template<typename T>
class fftconvolve_multiplyCreateKernel {
   public:
    fftconvolve_multiplyCreateKernel(write_accessor<T> d_out, KParam oInfo,
                                     read_accessor<T> d_in1, KParam i1Info,
                                     read_accessor<T> d_in2, KParam i2Info,
                                     const int nelem, const int kind)
        : d_out_(d_out)
        , oInfo_(oInfo)
        , d_in1_(d_in1)
        , i1Info_(i1Info)
        , d_in2_(d_in2)
        , i2Info_(i2Info)
        , nelem_(nelem)
        , kind_(kind) {}
    void operator()(sycl::nd_item<1> it) const {
        const int t = it.get_global_id(0);

        if (t >= nelem_) return;

        if (kind_ == AF_BATCH_NONE || kind_ == AF_BATCH_SAME) {
            // Complex multiply each signal to equivalent filter
            const int ridx = t * 2;
            const int iidx = t * 2 + 1;

            T a = d_in1_[i1Info_.offset + ridx];
            T b = d_in1_[i1Info_.offset + iidx];
            T c = d_in2_[i2Info_.offset + ridx];
            T d = d_in2_[i2Info_.offset + iidx];

            d_out_[oInfo_.offset + ridx] = a * c - b * d;
            d_out_[oInfo_.offset + iidx] = a * d + b * c;
        } else if (kind_ == AF_BATCH_LHS) {
            // Complex multiply all signals to filter
            const int ridx1 = t * 2;
            const int iidx1 = t * 2 + 1;

            // Treating complex output array as real-only array,
            // thus, multiply strides by 2
            const int ridx2 =
                ridx1 % (i2Info_.strides[3] * i2Info_.dims[3] * 2);
            const int iidx2 =
                iidx1 % (i2Info_.strides[3] * i2Info_.dims[3] * 2);

            T a = d_in1_[i1Info_.offset + ridx1];
            T b = d_in1_[i1Info_.offset + iidx1];
            T c = d_in2_[i2Info_.offset + ridx2];
            T d = d_in2_[i2Info_.offset + iidx2];

            d_out_[oInfo_.offset + ridx1] = a * c - b * d;
            d_out_[oInfo_.offset + iidx1] = a * d + b * c;
        } else if (kind_ == AF_BATCH_RHS) {
            // Complex multiply signal to all filters
            const int ridx2 = t * 2;
            const int iidx2 = t * 2 + 1;

            // Treating complex output array as real-only array,
            // thus, multiply strides by 2
            const int ridx1 =
                ridx2 % (i1Info_.strides[3] * i1Info_.dims[3] * 2);
            const int iidx1 =
                iidx2 % (i1Info_.strides[3] * i1Info_.dims[3] * 2);

            T a = d_in1_[i1Info_.offset + ridx1];
            T b = d_in1_[i1Info_.offset + iidx1];
            T c = d_in2_[i2Info_.offset + ridx2];
            T d = d_in2_[i2Info_.offset + iidx2];

            d_out_[oInfo_.offset + ridx2] = a * c - b * d;
            d_out_[oInfo_.offset + iidx2] = a * d + b * c;
        }
    }

   private:
    write_accessor<T> d_out_;
    KParam oInfo_;
    read_accessor<T> d_in1_;
    KParam i1Info_;
    read_accessor<T> d_in2_;
    KParam i2Info_;
    const int nelem_;
    const int kind_;
};

template<typename convT, typename T>
void complexMultiplyHelper(Param<convT> packed, Param<T> sig, Param<T> filter,
                           const int rank, AF_BATCH_KIND kind) {
    Param<T> sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, rank, kind);

    int sig_packed_elem = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];
    int filter_packed_elem =
        filter_tmp.info.strides[3] * filter_tmp.info.dims[3];
    int mul_elem = (sig_packed_elem < filter_packed_elem) ? filter_packed_elem
                                                          : sig_packed_elem;
    int blocks   = divup(mul_elem, THREADS);

    auto local  = sycl::range(THREADS);
    auto global = sycl::range(blocks * THREADS);

    // Treat complex output as an array of scalars
    using convScalarT      = typename convT::value_type;
    auto packed_num_elem   = (*packed.data).get_range().size();
    auto packed_tmp_buffer = (*packed.data)
                                 .template reinterpret<convScalarT>(
                                     sycl::range<1>{packed_num_elem * 2});
    auto sig_tmp_buffer = (*packed.data)
                              .template reinterpret<convScalarT>(
                                  sycl::range<1>{packed_num_elem * 2});
    auto filter_tmp_buffer = (*packed.data)
                                 .template reinterpret<convScalarT>(
                                     sycl::range<1>{packed_num_elem * 2});

    getQueue().submit([&](auto &h) {
        write_accessor<convScalarT> d_packed    = {packed_tmp_buffer, h};
        read_accessor<convScalarT> d_sig_tmp    = {sig_tmp_buffer, h};
        read_accessor<convScalarT> d_filter_tmp = {filter_tmp_buffer, h};
        h.parallel_for(
            sycl::nd_range{global, local},
            fftconvolve_multiplyCreateKernel<typename convT::value_type>(
                d_packed, packed.info, d_sig_tmp, sig_tmp.info, d_filter_tmp,
                filter_tmp.info, mul_elem, (int)kind));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
