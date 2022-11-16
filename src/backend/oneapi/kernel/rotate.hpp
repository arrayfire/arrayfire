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
#include <types.hpp>
#include <common/complex.hpp>
#include <kernel/interp.hpp>

#include <string>
#include <vector>

using af::dtype_traits;

namespace oneapi {
namespace kernel {

template<typename T>
struct itype_t {
    typedef float wtype;
    typedef float vtype;
};

template<>
struct itype_t<double> {
    typedef double wtype;
    typedef double vtype;
};

template<>
struct itype_t<cfloat> {
    typedef float wtype;
    typedef cfloat vtype;
};

template<>
struct itype_t<cdouble> {
    typedef double wtype;
    typedef cdouble vtype;
};

typedef struct {
    float tmat[6];
} tmat_t;

template<typename T>
using local_accessor = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                      sycl::access::target::local>;
template<typename T>
using read_accessor = sycl::accessor<T, 1, sycl::access::mode::read>;
template<typename T>
using write_accessor = sycl::accessor<T, 1, sycl::access::mode::write>;

template<typename T>
using wtype_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;

template<typename T>
using vtype_t = typename std::conditional<common::is_complex<T>::value, T,
                                          wtype_t<T>>::type;

template<typename InterpInTy, int INTERP_ORDER>
class rotateExtractKernel {
public:
  rotateExtractKernel(write_accessor<InterpInTy> d_out, const KParam out,
                      read_accessor<InterpInTy> d_in, const KParam in,
                      const tmat_t t, const int nimages, const int batches,
                      const int blocksXPerImage, const int blocksYPerImage,
                      af::interpType method) // , sycl::accessor<float, 2> debugAcc)
    : d_out_(d_out), out_(out), d_in_(d_in), in_(in), t_(t), nimages_(nimages), batches_(batches), blocksXPerImage_(blocksXPerImage), blocksYPerImage_(blocksYPerImage), method_(method) {} // , debugAcc_(debugAcc) {}
  void operator()(sycl::nd_item<2> it) const {
      // debugAcc_[it.get_global_id(0)][it.get_global_id(1)] = -1;

      using BT = typename dtype_traits<InterpInTy>::base_type;
      using InterpPosTy = typename dtype_traits<wtype_t<BT>>::base_type;

      auto g = it.get_group();

      // Compute which image set
      const int setId      = g.get_group_id(0) / blocksXPerImage_;
      const int blockIdx_x = g.get_group_id(0) - setId * blocksXPerImage_;

      const int batch      = g.get_group_id(1) / blocksYPerImage_;
      const int blockIdx_y = g.get_group_id(1) - batch * blocksYPerImage_;

      // Get thread indices
      const int xido = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
      const int yido = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

      const int limages = fmin((int)out_.dims[2] - setId * nimages_, nimages_);

      if (xido >= out_.dims[0] || yido >= out_.dims[1]) return;

      InterpPosTy xidi = xido * t_.tmat[0] + yido * t_.tmat[1] + t_.tmat[2];
      InterpPosTy yidi = xido * t_.tmat[3] + yido * t_.tmat[4] + t_.tmat[5];

      int outoff =
        out_.offset + setId * nimages_ * out_.strides[2] + batch * out_.strides[3];
      int inoff =
        in_.offset + setId * nimages_ * in_.strides[2] + batch * in_.strides[3];

      const int loco = outoff + (yido * out_.strides[1] + xido);

      InterpInTy zero = (InterpInTy)0;
      if (INTERP_ORDER > 1) {
        // Special conditions to deal with boundaries for bilinear and bicubic
        // FIXME: Ideally this condition should be removed or be present for all
        // methods But tests are expecting a different behavior for bilinear and
        // nearest
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
      const bool doclamp = INTERP_ORDER != 1;
      constexpr int XDIM = 0, YDIM = 1;
      Interp2<InterpInTy, InterpPosTy, INTERP_ORDER> interp;
      const int nelems = out_.dims[0] * out_.dims[1] * out_.dims[2];
      // debugAcc_[it.get_global_id(0)][it.get_global_id(1)] = loco < nelems;
      interp(d_out_, out_, loco, d_in_, in_, inoff, xidi, yidi, XDIM, YDIM, method_, limages, doclamp, 2);
  }
private:
write_accessor<InterpInTy> d_out_;
const KParam out_;
read_accessor<InterpInTy> d_in_;
const KParam in_;
const tmat_t t_;
const int nimages_;
const int batches_;
const int blocksXPerImage_;
const int blocksYPerImage_;
af::interpType method_;
// sycl::accessor<float, 2> debugAcc_;
};

#include "/home/gpryor/new-dev/io.hpp"
#include "/home/gpryor/new-dev/msg.hpp"

template<typename T>
void rotate(Param<T> out, Param<T> in, float theta, af_interp_type method, int order) {
// #ifndef DO_NOT_WRITE
//     OPEN_W("/home/gpryor/new-dev/data/test-00");
//     WRITE(out); WRITE(in); WRITE(theta); WRITE(method); WRITE(order);
// #endif

    constexpr int TX = 16;
    constexpr int TY = 16;
    // Used for batching images
    constexpr int TI = 4;
    constexpr bool isComplex =
        static_cast<af_dtype>(dtype_traits<T>::af_type) == c32 ||
        static_cast<af_dtype>(dtype_traits<T>::af_type) == c64;

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

    auto local = sycl::range{TX, TY};

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

    // sycl::buffer<float, 2> debugBuffer{sycl::range(global_x, global_y)};

    getQueue().submit([&](sycl::handler &h) {
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor inAcc{*in.data, h, sycl::read_only};
        // sycl::accessor<float, 2> debugAcc{debugBuffer, h};
        switch (order) {
        case 1:
          h.parallel_for(sycl::nd_range{global, local},
                         rotateExtractKernel<T, 1>(outAcc, out.info, inAcc, in.info, t, nimages,
                                                   nbatches, blocksXPerImage, blocksYPerImage, method));
                         // debugAcc));
          break;
        // case 2:
        //   h.parallel_for(sycl::nd_range{global, local},
        //                  rotateExtractKernel<T, 2>(outAcc, out.info, inAcc, in.info, t, nimages,
        //                                            nbatches, blocksXPerImage, blocksYPerImage, method));
        //   break;
        // case 3:
        //   h.parallel_for(sycl::nd_range{global, local},
        //                  rotateExtractKernel<T, 3>(outAcc, out.info, inAcc, in.info, t, nimages,
        //                                            nbatches, blocksXPerImage, blocksYPerImage, method));
        //   break;
        // default:
        //   ONEAPI_NOT_SUPPORTED("invalid interpolation order");
        }
    }).wait();

    // M(debugBuffer);

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
