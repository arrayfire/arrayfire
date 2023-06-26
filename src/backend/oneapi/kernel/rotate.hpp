/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/interp.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

typedef struct {
    float tmat[6];
} tmat_t;

template<typename T>
using wtype_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;

template<typename T>
using vtype_t = typename std::conditional<common::is_complex<T>::value, T,
                                          wtype_t<T>>::type;

template<typename T, typename InterpInTy, typename InterpPosTy,
         int INTERP_ORDER>
class rotateCreateKernel {
   public:
    rotateCreateKernel(write_accessor<T> d_out, const KParam out,
                       read_accessor<T> d_in, const KParam in, const tmat_t t,
                       const int nimages, const int batches,
                       const int blocksXPerImage, const int blocksYPerImage,
                       af::interpType method)
        : d_out_(d_out)
        , out_(out)
        , d_in_(d_in)
        , in_(in)
        , t_(t)
        , nimages_(nimages)
        , batches_(batches)
        , blocksXPerImage_(blocksXPerImage)
        , blocksYPerImage_(blocksYPerImage)
        , method_(method) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        // Compute which image set
        const int setId      = g.get_group_id(0) / blocksXPerImage_;
        const int blockIdx_x = g.get_group_id(0) - setId * blocksXPerImage_;

        const int batch      = g.get_group_id(1) / blocksYPerImage_;
        const int blockIdx_y = g.get_group_id(1) - batch * blocksYPerImage_;

        // Get thread indices
        const int xido = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
        const int yido = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

        const int limages =
            std::min((int)out_.dims[2] - setId * nimages_, nimages_);

        if (xido >= (unsigned)out_.dims[0] || yido >= (unsigned)out_.dims[1])
            return;

        InterpPosTy xidi = xido * t_.tmat[0] + yido * t_.tmat[1] + t_.tmat[2];
        InterpPosTy yidi = xido * t_.tmat[3] + yido * t_.tmat[4] + t_.tmat[5];

        int outoff = out_.offset + setId * nimages_ * out_.strides[2] +
                     batch * out_.strides[3];
        int inoff = in_.offset + setId * nimages_ * in_.strides[2] +
                    batch * in_.strides[3];

        const int loco = outoff + (yido * out_.strides[1] + xido);

        InterpInTy zero = (InterpInTy)0;
        if constexpr (INTERP_ORDER > 1) {
            // Special conditions to deal with boundaries for bilinear and
            // bicubic
            // FIXME: Ideally this condition should be removed or be present for
            // all  methods But tests are expecting a different behavior for
            // bilinear and nearest
            if (xidi < (InterpPosTy)-0.0001 || yidi < (InterpPosTy)-0.0001 ||
                in_.dims[0] <= xidi || in_.dims[1] <= yidi) {
                for (int i = 0; i < nimages_; i++) {
                    d_out_[loco + i * out_.strides[2]] = zero;
                }
                return;
            }
        }

        // FIXME: Nearest and lower do not do clamping, but other methods do
        // Make it consistent
        constexpr bool doclamp = INTERP_ORDER != 1;
        Interp2<T, InterpPosTy, INTERP_ORDER> interp2;
        interp2(d_out_, out_, loco, d_in_, in_, inoff, xidi, yidi, 0, 1,
                method_, limages, doclamp, 2);
    }

   private:
    write_accessor<T> d_out_;
    const KParam out_;
    read_accessor<T> d_in_;
    const KParam in_;
    const tmat_t t_;
    const int nimages_;
    const int batches_;
    const int blocksXPerImage_;
    const int blocksYPerImage_;
    af::interpType method_;
};

template<typename T>
void rotate(Param<T> out, const Param<T> in, const float theta,
            af_interp_type method, int order) {
    using std::string;

    using BT = typename dtype_traits<T>::base_type;

    constexpr int TX = 16;
    constexpr int TY = 16;

    // Used for batching images
    constexpr int TI = 4;

    const float c = cos(-theta), s = sin(-theta);
    float tx, ty;
    {
        const float nx = 0.5 * (in.info.dims[0] - 1);
        const float ny = 0.5 * (in.info.dims[1] - 1);
        const float mx = 0.5 * (out.info.dims[0] - 1);
        const float my = 0.5 * (out.info.dims[1] - 1);
        const float sx = (mx * c + my * -s);
        const float sy = (mx * s + my * c);
        tx             = -(sx - nx);
        ty             = -(sy - ny);
    }

    // Rounding error. Anything more than 3 decimal points wont make a diff
    tmat_t t;
    t.tmat[0] = round(c * 1000) / 1000.0f;
    t.tmat[1] = round(-s * 1000) / 1000.0f;
    t.tmat[2] = round(tx * 1000) / 1000.0f;
    t.tmat[3] = round(s * 1000) / 1000.0f;
    t.tmat[4] = round(c * 1000) / 1000.0f;
    t.tmat[5] = round(ty * 1000) / 1000.0f;

    auto local = sycl::range(TX, TY);

    int nimages               = in.info.dims[2];
    int nbatches              = in.info.dims[3];
    int global_x              = local[0] * divup(out.info.dims[0], local[0]);
    int global_y              = local[1] * divup(out.info.dims[1], local[1]);
    const int blocksXPerImage = global_x / local[0];
    const int blocksYPerImage = global_y / local[1];

    if (nimages > TI) {
        int tile_images = divup(nimages, TI);
        nimages         = TI;
        global_x        = global_x * tile_images;
    }
    global_y *= nbatches;

    auto global = sycl::range(global_x, global_y);

    getQueue().submit([&](auto &h) {
        read_accessor<T> d_in{*in.data, h};
        write_accessor<T> d_out{*out.data, h};
        switch (order) {
            case 1:
                h.parallel_for(
                    sycl::nd_range{global, local},
                    rotateCreateKernel<T, T, wtype_t<BT>, 1>(
                        d_out, out.info, d_in, in.info, t, nimages, nbatches,
                        blocksXPerImage, blocksYPerImage, method));
                break;
            case 2:
                h.parallel_for(
                    sycl::nd_range{global, local},
                    rotateCreateKernel<T, T, wtype_t<BT>, 2>(
                        d_out, out.info, d_in, in.info, t, nimages, nbatches,
                        blocksXPerImage, blocksYPerImage, method));
                break;
            case 3:
                h.parallel_for(
                    sycl::nd_range{global, local},
                    rotateCreateKernel<T, T, wtype_t<BT>, 3>(
                        d_out, out.info, d_in, in.info, t, nimages, nbatches,
                        blocksXPerImage, blocksYPerImage, method));
                break;
            default: throw std::string("invalid interpolation order");
        }
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
