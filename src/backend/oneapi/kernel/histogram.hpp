/*******************************************************
 * Copyright (c) 2022 ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

#define MAX_BINS 4000
#define THREADS_X 256
#define THRD_LOAD 16

// using memory_order = memory_order;
// using memory_scope = memory_scope;

template<typename T>
using local_atomic_ref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;

template<typename T>
using global_atomic_ref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

template<typename T, int dimensions>
using local_accessor =
    sycl::accessor<T, dimensions, sycl::access::mode::read_write,
                   sycl::access::target::local>;

template<typename T>
class histogramKernel {
   public:
    histogramKernel(sycl::accessor<uint> d_dst, KParam oInfo,
                    const sycl::accessor<T> d_src, KParam iInfo,
                    local_accessor<uint, 1> localMemAcc, int len, int nbins,
                    float minval, float maxval, int nBBS, const bool isLinear)
        : d_dst_(d_dst)
        , oInfo_(oInfo)
        , d_src_(d_src)
        , iInfo_(iInfo)
        , localMemAcc_(localMemAcc)
        , len_(len)
        , nbins_(nbins)
        , minval_(minval)
        , maxval_(maxval)
        , nBBS_(nBBS)
        , isLinear_(isLinear) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        unsigned b2   = g.get_group_id(0) / nBBS_;
        int start     = (g.get_group_id(0) - b2 * nBBS_) * THRD_LOAD *
                        g.get_local_range(0) +
                    it.get_local_id(0);
        int end = fmin((int)(start + THRD_LOAD * g.get_local_range(0)), len_);

        // offset input and output to account for batch ops
        const T *in = d_src_.get_pointer() + b2 * iInfo_.strides[2] +
                      g.get_group_id(1) * iInfo_.strides[3] + iInfo_.offset;
        uint outOffset =
            b2 * oInfo_.strides[2] + g.get_group_id(1) * oInfo_.strides[3];

        float dx = (maxval_ - minval_) / (float)nbins_;

        bool use_global = nbins_ > MAX_BINS;

        if (!use_global) {
            for (int i = it.get_local_id(0); i < nbins_;
                 i += g.get_local_range(0))
                localMemAcc_[i] = 0;
            it.barrier();
        }

        for (int row = start; row < end; row += g.get_local_range(0)) {
            const int i0  = row % iInfo_.dims[0];
            const int i1  = row / iInfo_.dims[0];
            const int idx = isLinear_ ? row : i0 + i1 * iInfo_.strides[1];

            int bin = (int)(((float)in[idx] - minval_) / dx);
            bin     = fmax(bin, 0);
            bin     = fmin(bin, (int)nbins_ - 1);

            if (use_global) {
                global_atomic_ref<uint>(d_dst_[outOffset + bin])++;
            } else {
                local_atomic_ref<uint>(localMemAcc_[bin])++;
            }
        }

        if (!use_global) {
            it.barrier();
            for (int i = it.get_local_id(0); i < nbins_;
                 i += g.get_local_range(0)) {
                global_atomic_ref<uint>(d_dst_[outOffset + i]) +=
                    localMemAcc_[i];
            }
        }
    }

   private:
    sycl::accessor<uint> d_dst_;
    KParam oInfo_;
    sycl::accessor<T> d_src_;
    KParam iInfo_;
    local_accessor<uint, 1> localMemAcc_;
    int len_;
    int nbins_;
    float minval_;
    float maxval_;
    int nBBS_;
    bool isLinear_;
};

template<typename T>
void histogram(Param<uint> out, const Param<T> in, int nbins, float minval,
               float maxval, bool isLinear) {
    int nElems  = in.info.dims[0] * in.info.dims[1];
    int blk_x   = divup(nElems, THRD_LOAD * THREADS_X);
    int locSize = nbins <= MAX_BINS ? (nbins * sizeof(uint)) : 1;

    auto local           = sycl::range{THREADS_X, 1};
    const size_t global0 = blk_x * in.info.dims[2] * THREADS_X;
    const size_t global1 = in.info.dims[3];
    auto global          = sycl::range{global0, global1};

    // \TODO drop this first memset once createEmptyArray is reverted back to
    //       createValueArray in ../histogram.cpp
    getQueue()
        .submit([&](sycl::handler &h) {
            auto outAcc = out.data->get_access(h);
            h.parallel_for(sycl::range<1>{(size_t)nbins},
                           [=](sycl::id<1> idx) { outAcc[idx[0]] = 0; });
        })
        .wait();
    getQueue().submit([&](sycl::handler &h) {
        auto inAcc  = in.data->get_access(h);
        auto outAcc = out.data->get_access(h);
        sycl::stream debugStream(128, 128, h);

        auto localMem = local_accessor<uint, 1>(locSize, h);

        h.parallel_for(
            sycl::nd_range{global, local},
            histogramKernel<T>(outAcc, out.info, inAcc, in.info, localMem,
                               nElems, nbins, minval, maxval, blk_x, isLinear));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
