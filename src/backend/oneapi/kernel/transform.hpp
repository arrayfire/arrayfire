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
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/interp.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
using wtype_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;

template<typename T>
using vtype_t = typename std::conditional<common::is_complex<T>::value, T,
                                          wtype_t<T>>::type;

template<bool PERSPECTIVE>
void calc_transf_inverse(float *txo, const float *txi) {
    if constexpr (PERSPECTIVE) {
        txo[0] = txi[4] * txi[8] - txi[5] * txi[7];
        txo[1] = -(txi[1] * txi[8] - txi[2] * txi[7]);
        txo[2] = txi[1] * txi[5] - txi[2] * txi[4];

        txo[3] = -(txi[3] * txi[8] - txi[5] * txi[6]);
        txo[4] = txi[0] * txi[8] - txi[2] * txi[6];
        txo[5] = -(txi[0] * txi[5] - txi[2] * txi[3]);

        txo[6] = txi[3] * txi[7] - txi[4] * txi[6];
        txo[7] = -(txi[0] * txi[7] - txi[1] * txi[6]);
        txo[8] = txi[0] * txi[4] - txi[1] * txi[3];

        float det = txi[0] * txo[0] + txi[1] * txo[3] + txi[2] * txo[6];

        txo[0] /= det;
        txo[1] /= det;
        txo[2] /= det;
        txo[3] /= det;
        txo[4] /= det;
        txo[5] /= det;
        txo[6] /= det;
        txo[7] /= det;
        txo[8] /= det;
    } else {
        float det = txi[0] * txi[4] - txi[1] * txi[3];

        txo[0] = txi[4] / det;
        txo[1] = txi[3] / det;
        txo[3] = txi[1] / det;
        txo[4] = txi[0] / det;

        txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
        txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
    }
}

template<typename T, typename InterpPosTy, bool PERSPECTIVE, int INTERP_ORDER>
class transformCreateKernel {
   public:
    transformCreateKernel(write_accessor<T> d_out, const KParam out,
                          read_accessor<T> d_in, const KParam in,
                          read_accessor<float> c_tmat, const KParam tf,
                          const int nImg2, const int nImg3, const int nTfs2,
                          const int nTfs3, const int batchImg2,
                          const int blocksXPerImage, const int blocksYPerImage,
                          const af::interpType method, const bool INVERSE)
        : d_out_(d_out)
        , out_(out)
        , d_in_(d_in)
        , in_(in)
        , c_tmat_(c_tmat)
        , tf_(tf)
        , nImg2_(nImg2)
        , nImg3_(nImg3)
        , nTfs2_(nTfs2)
        , nTfs3_(nTfs3)
        , batchImg2_(batchImg2)
        , blocksXPerImage_(blocksXPerImage)
        , blocksYPerImage_(blocksYPerImage)
        , method_(method)
        , INVERSE_(INVERSE) {}
    void operator()(sycl::nd_item<3> it) const {
        sycl::group g = it.get_group();

        // Image Ids
        const int imgId2 = g.get_group_id(0) / blocksXPerImage_;
        const int imgId3 = g.get_group_id(1) / blocksYPerImage_;

        // Block in_ local image
        const int blockIdx_x = g.get_group_id(0) - imgId2 * blocksXPerImage_;
        const int blockIdx_y = g.get_group_id(1) - imgId3 * blocksYPerImage_;

        // Get thread indices in_ local image
        const int xido = blockIdx_x * g.get_local_range(0) + it.get_local_id(0);
        const int yido = blockIdx_y * g.get_local_range(1) + it.get_local_id(1);

        // Image iteration loop count for image batching
        int limages = sycl::min(
            sycl::max((int)(out_.dims[2] - imgId2 * nImg2_), 1), batchImg2_);

        if (xido >= out_.dims[0] || yido >= out_.dims[1]) return;

        // Index of transform
        const int eTfs2 = sycl::max((nTfs2_ / nImg2_), 1);

        int t_idx3        = -1;  // init
        int t_idx2        = -1;  // init
        int t_idx2_offset = 0;

        const int blockIdx_z = g.get_group_id(2);

        if (nTfs3_ == 1) {
            t_idx3 = 0;  // Always 0 as only 1 transform defined
        } else {
            if (nTfs3_ == nImg3_) {
                t_idx3 =
                    imgId3;  // One to one batch with all transforms defined
            } else {
                t_idx3 = blockIdx_z / eTfs2;  // Transform batched, calculate
                t_idx2_offset = t_idx3 * nTfs2_;
            }
        }

        if (nTfs2_ == 1) {
            t_idx2 = 0;  // Always 0 as only 1 transform defined
        } else {
            if (nTfs2_ == nImg2_) {
                t_idx2 =
                    imgId2;  // One to one batch with all transforms defined
            } else {
                t_idx2 =
                    blockIdx_z - t_idx2_offset;  // Transform batched, calculate
            }
        }

        // Linear transform index
        const int t_idx = t_idx2 + t_idx3 * nTfs2_;

        // Global outoff
        int outoff = out_.offset;
        int inoff  = imgId2 * batchImg2_ * in_.strides[2] +
                    imgId3 * in_.strides[3] + in_.offset;
        if (nImg2_ == nTfs2_ || nImg2_ > 1) {  // One-to-One or Image on dim2
            outoff += imgId2 * batchImg2_ * out_.strides[2];
        } else {  // Transform batched on dim2
            outoff += t_idx2 * out_.strides[2];
        }

        if (nImg3_ == nTfs3_ || nImg3_ > 1) {  // One-to-One or Image on dim3
            outoff += imgId3 * out_.strides[3];
        } else {  // Transform batched on dim2
            outoff += t_idx3 * out_.strides[3];
        }

        // Transform is in_ global memory.
        // Needs outoff to correct transform being processed.
        const int transf_len = PERSPECTIVE ? 9 : 6;
        using TMatTy =
            typename std::conditional<PERSPECTIVE, float[9], float[6]>::type;
        TMatTy tmat;
        const float *tmat_ptr = c_tmat_.get_pointer() + t_idx * transf_len;

        // We expect a inverse transform matrix by default
        // If it is an forward transform, then we need its inverse
        if (INVERSE_ == 1) {
#pragma unroll 3
            for (int i = 0; i < transf_len; i++) tmat[i] = tmat_ptr[i];
        } else {
            calc_transf_inverse<PERSPECTIVE>(tmat, tmat_ptr);
        }

        InterpPosTy xidi = xido * tmat[0] + yido * tmat[1] + tmat[2];
        InterpPosTy yidi = xido * tmat[3] + yido * tmat[4] + tmat[5];

        if constexpr (PERSPECTIVE) {
            const InterpPosTy W = xido * tmat[6] + yido * tmat[7] + tmat[8];
            xidi /= W;
            yidi /= W;
        }
        const int loco = outoff + (yido * out_.strides[1] + xido);
        // FIXME: Nearest and lower do not do clamping, but other methods do
        // Make it consistent
        const bool doclamp = INTERP_ORDER != 1;

        T zero = (T)0;
        if (xidi < (InterpPosTy)-0.0001f || yidi < (InterpPosTy)-0.0001f ||
            in_.dims[0] <= xidi || in_.dims[1] <= yidi) {
            for (int n = 0; n < limages; n++) {
                d_out_[loco + n * out_.strides[2]] = zero;
            }
            return;
        }

        Interp2<T, InterpPosTy, INTERP_ORDER> interp2;
        interp2(d_out_, out_, loco, d_in_, in_, inoff, xidi, yidi, 0, 1,
                method_, limages, doclamp, 2);
    }

   private:
    write_accessor<T> d_out_;
    const KParam out_;
    read_accessor<T> d_in_;
    const KParam in_;
    read_accessor<float> c_tmat_;
    const KParam tf_;
    const int nImg2_;
    const int nImg3_;
    const int nTfs2_;
    const int nTfs3_;
    const int batchImg2_;
    const int blocksXPerImage_;
    const int blocksYPerImage_;
    const af::interpType method_;
    const bool INVERSE_;
};

template<typename T>
void transform(Param<T> out, const Param<T> in, const Param<float> tf,
               bool isInverse, bool isPerspective, af_interp_type method,
               int order) {
    using std::string;

    using BT = typename dtype_traits<T>::base_type;

    constexpr int TX = 16;
    constexpr int TY = 16;
    // Used for batching images
    constexpr int TI = 4;

    const int nImg2 = in.info.dims[2];
    const int nImg3 = in.info.dims[3];
    const int nTfs2 = tf.info.dims[2];
    const int nTfs3 = tf.info.dims[3];

    auto local = sycl::range(TX, TY, 1);

    int batchImg2 = 1;
    if (nImg2 != nTfs2) batchImg2 = fmin(nImg2, TI);

    const int blocksXPerImage = divup(out.info.dims[0], local[0]);
    const int blocksYPerImage = divup(out.info.dims[1], local[1]);

    int global_x = local[0] * blocksXPerImage * (nImg2 / batchImg2);
    int global_y = local[1] * blocksYPerImage * nImg3;
    int global_z =
        local[2] * fmax((nTfs2 / nImg2), 1) * fmax((nTfs3 / nImg3), 1);

    auto global = sycl::range(global_x, global_y, global_z);

#define INVOKE(PERSPECTIVE, INTERP_ORDER)                                      \
    h.parallel_for(                                                            \
        sycl::nd_range{global, local},                                         \
        transformCreateKernel<T, wtype_t<BT>, PERSPECTIVE, INTERP_ORDER>(      \
            d_out, out.info, d_in, in.info, d_tf, tf.info, nImg2, nImg3,       \
            nTfs2, nTfs3, batchImg2, blocksXPerImage, blocksYPerImage, method, \
            isInverse));

    getQueue().submit([&](auto &h) {
        read_accessor<T> d_in{*in.data, h};
        read_accessor<float> d_tf{*tf.data, h};
        write_accessor<T> d_out{*out.data, h};

        if (isPerspective == true && order == 1) INVOKE(true, 1);
        if (isPerspective == true && order == 2) INVOKE(true, 2);
        if (isPerspective == true && order == 3) INVOKE(true, 3);

        if (isPerspective == false && order == 1) INVOKE(false, 1);
        if (isPerspective == false && order == 2) INVOKE(false, 2);
        if (isPerspective == false && order == 3) INVOKE(false, 3);
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
