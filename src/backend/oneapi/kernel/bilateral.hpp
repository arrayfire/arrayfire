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
#include <kernel/accessors.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename outType, bool USE_NATIVE_EXP>
auto exp_native_nonnative(float in) {
    if constexpr (USE_NATIVE_EXP)
        return sycl::native::exp(in);
    else
        return exp(in);
}

template<typename outType, typename inType, bool USE_NATIVE_EXP>
class bilateralKernel {
   public:
    bilateralKernel(write_accessor<outType> d_dst, KParam oInfo,
                    read_accessor<inType> d_src, KParam iInfo,
                    sycl::local_accessor<outType, 1> localMem,
                    sycl::local_accessor<outType, 1> gauss2d, float sigma_space,
                    float sigma_color, int gaussOff, int nBBS0, int nBBS1)
        : d_dst_(d_dst)
        , oInfo_(oInfo)
        , d_src_(d_src)
        , iInfo_(iInfo)
        , localMem_(localMem)
        , gauss2d_(gauss2d)
        , sigma_space_(sigma_space)
        , sigma_color_(sigma_color)
        , gaussOff_(gaussOff)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g              = it.get_group();
        const int radius           = sycl::max((int)(sigma_space_ * 1.5f), 1);
        const int padding          = 2 * radius;
        const int window_size      = padding + 1;
        const int shrdLen          = g.get_local_range(0) + padding;
        const float variance_range = sigma_color_ * sigma_color_;
        const float variance_space = sigma_space_ * sigma_space_;
        const float variance_space_neg2     = -2.0 * variance_space;
        const float inv_variance_range_neg2 = -0.5 / (variance_range);

        // gfor batch offsets
        unsigned b2 = g.get_group_id(0) / nBBS0_;
        unsigned b3 = g.get_group_id(1) / nBBS1_;

        const inType* in =
            d_src_.get_pointer() +
            (b2 * iInfo_.strides[2] + b3 * iInfo_.strides[3] + iInfo_.offset);
        outType* out = d_dst_.get_pointer() +
                       (b2 * oInfo_.strides[2] + b3 * oInfo_.strides[3]);

        int lx = it.get_local_id(0);
        int ly = it.get_local_id(1);

        const int gx =
            g.get_local_range(0) * (g.get_group_id(0) - b2 * nBBS0_) + lx;
        const int gy =
            g.get_local_range(1) * (g.get_group_id(1) - b3 * nBBS1_) + ly;

        // generate gauss2d_ spatial variance values for block
        if (lx < window_size && ly < window_size) {
            int x = lx - radius;
            int y = ly - radius;
            gauss2d_[ly * window_size + lx] =
                exp_native_nonnative<outType, USE_NATIVE_EXP>(
                    ((x * x) + (y * y)) / variance_space_neg2);
        }

        int s0 = iInfo_.strides[0];
        int s1 = iInfo_.strides[1];
        int d0 = iInfo_.dims[0];
        int d1 = iInfo_.dims[1];
        // pull image to local memory
        for (int b = ly, gy2 = gy; b < shrdLen;
             b += g.get_local_range(1), gy2 += g.get_local_range(1)) {
            // move row_set g.get_local_range(1) along coloumns
            for (int a = lx, gx2 = gx; a < shrdLen;
                 a += g.get_local_range(0), gx2 += g.get_local_range(0)) {
                load2LocalMem(localMem_, in, a, b, shrdLen, d0, d1,
                              gx2 - radius, gy2 - radius, s1, s0);
            }
        }

        it.barrier();

        if (gx < iInfo_.dims[0] && gy < iInfo_.dims[1]) {
            lx += radius;
            ly += radius;
            outType center_color = localMem_[ly * shrdLen + lx];
            outType res          = 0;
            outType norm         = 0;

            int joff = (ly - radius) * shrdLen + (lx - radius);
            int goff = 0;

            for (int wj = 0; wj < window_size; ++wj) {
                for (int wi = 0; wi < window_size; ++wi) {
                    outType tmp_color = localMem_[joff + wi];
                    const outType c   = center_color - tmp_color;
                    outType gauss_range =
                        exp_native_nonnative<outType, USE_NATIVE_EXP>(
                            c * c * inv_variance_range_neg2);
                    outType weight = gauss2d_[goff + wi] * gauss_range;
                    norm += weight;
                    res += tmp_color * weight;
                }
                joff += shrdLen;
                goff += window_size;
            }
            out[gy * oInfo_.strides[1] + gx] = res / norm;
        }
    }

    int lIdx(int x, int y, int stride1, int stride0) const {
        return (y * stride1 + x * stride0);
    }

    template<class T>
    constexpr const T& clamp0(const T& v, const T& lo, const T& hi) const {
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    void load2LocalMem(sycl::local_accessor<outType, 1> shrd, const inType* in,
                       int lx, int ly, int shrdStride, int dim0, int dim1,
                       int gx, int gy, int inStride1, int inStride0) const {
        int gx_ = sycl::clamp(gx, 0, dim0 - 1);
        int gy_ = sycl::clamp(gy, 0, dim1 - 1);
        shrd[lIdx(lx, ly, shrdStride, 1)] =
            (outType)in[lIdx(gx_, gy_, inStride1, inStride0)];
    }

   private:
    write_accessor<outType> d_dst_;
    KParam oInfo_;
    read_accessor<inType> d_src_;
    KParam iInfo_;
    sycl::local_accessor<outType, 1> localMem_;
    sycl::local_accessor<outType, 1> gauss2d_;
    float sigma_space_;
    float sigma_color_;
    int gaussOff_;
    int nBBS0_;
    int nBBS1_;
};

template<typename inType, typename outType>
void bilateral(Param<outType> out, const Param<inType> in, const float s_sigma,
               const float c_sigma) {
    constexpr int THREADS_X     = 16;
    constexpr int THREADS_Y     = 16;
    constexpr bool UseNativeExp = !std::is_same<inType, double>::value ||
                                  std::is_same<inType, cdouble>::value;

    auto local = sycl::range{THREADS_X, THREADS_Y};

    const int blk_x = divup(in.info.dims[0], THREADS_X);
    const int blk_y = divup(in.info.dims[1], THREADS_Y);

    auto global = sycl::range{(size_t)(blk_x * in.info.dims[2] * THREADS_X),
                              (size_t)(blk_y * in.info.dims[3] * THREADS_Y)};

    // calculate local memory size
    int radius          = (int)std::max(s_sigma * 1.5f, 1.f);
    int num_shrd_elems  = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    int num_gauss_elems = (2 * radius + 1) * (2 * radius + 1);
    size_t localMemSize = (num_shrd_elems + num_gauss_elems) * sizeof(outType);
    size_t MaxLocalSize =
        getQueue().get_device().get_info<sycl::info::device::local_mem_size>();
    if (localMemSize > MaxLocalSize) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nOneAPI Bilateral filter doesn't support %f spatial sigma\n",
                 s_sigma);
        ONEAPI_NOT_SUPPORTED(errMessage);
    }

    getQueue().submit([&](sycl::handler& h) {
        read_accessor<inType> inAcc{*in.data, h};
        write_accessor<outType> outAcc{*out.data, h};

        auto localMem = sycl::local_accessor<outType, 1>(num_shrd_elems, h);
        auto gauss2d  = sycl::local_accessor<outType, 1>(num_shrd_elems, h);

        h.parallel_for(sycl::nd_range{global, local},
                       bilateralKernel<outType, inType, UseNativeExp>(
                           outAcc, out.info, inAcc, in.info, localMem, gauss2d,
                           s_sigma, c_sigma, num_shrd_elems, blk_x, blk_y));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
