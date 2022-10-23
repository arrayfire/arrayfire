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

template<typename T, int dimensions>
using local_accessor =
    sycl::accessor<T, dimensions, sycl::access::mode::read_write,
                   sycl::access::target::local>;

template<typename T>
void rotate(Param<T> out, Param<T> in, float theta, af_interp_type method,
            int order) {
    constexpr int TX = 4;
    constexpr int TY = 4;
    constexpr int TI = 4;  // Used for batching images
    constexpr bool isComplex =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type) == c32 ||
        static_cast<af_dtype>(af::dtype_traits<T>::af_type) == c64;

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

    const float tmat[6] = {
        (float)round(c * 1000) / 1000.0f,  (float)round(-s * 1000) / 1000.0f,
        (float)round(tx * 1000) / 1000.0f, (float)round(s * 1000) / 1000.0f,
        (float)round(c * 1000) / 1000.0f,  (float)round(ty * 1000) / 1000.0f};

    const auto local = sycl::range{TX, TY};

    size_t nimages        = in.info.dims[2];
    const size_t nbatches = in.info.dims[3];

    size_t global_x              = local[0] * divup(out.info.dims[0], local[0]);
    size_t global_y              = local[1] * divup(out.info.dims[1], local[1]);
    const size_t blocksXPerImage = global_x / local[0];
    const size_t blocksYPerImage = global_y / local[1];

    if (nimages > TI) {
        const size_t tile_images = divup(nimages, TI);
        nimages                  = TI;
        global_x                 = global_x * tile_images;
    }
    global_y *= nbatches;

    const auto global = sycl::range{global_x, global_y};

    sycl::buffer<float, 2> debugBuffer{sycl::range<2>{global[0], global[1]}};

    getQueue()
        .submit([&](sycl::handler &h) {
            auto debugBufferAcc = debugBuffer.get_access(h);
            h.parallel_for(
                sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
                    debugBufferAcc[it.get_global_id(0)][it.get_global_id(1)] =
                        0;
                });
        })
        .wait();

    getQueue()
        .submit([&](sycl::handler &h) {
            auto d_out          = out.data->get_access(h);
            auto d_in           = in.data->get_access(h);
            auto debugBufferAcc = debugBuffer.get_access(h);

            h.parallel_for(
                sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
                    sycl::group g = it.get_group();

                    // Compute which image set
                    const int setId = g.get_group_id(0) / blocksXPerImage;
                    const int blockIdx_x =
                        g.get_group_id(0) - setId * blocksXPerImage;

                    const int batch = g.get_group_id(1) / blocksYPerImage;
                    const int blockIdx_y =
                        g.get_group_id(1) - batch * blocksYPerImage;

                    // Get thread indices
                    const int xido = it.get_global_id(0);
                    const int yido = it.get_global_id(1);

                    const int limages =
                        fmin((int)out.info.dims[2] - setId * nimages, nimages);

                    if (xido >= out.info.dims[0] || yido >= out.info.dims[1])
                        return;

                    // Compute input index
                    typedef typename itype_t<T>::wtype WT;
                    WT xidi = xido * tmat[0] + yido * tmat[1] + tmat[2];
                    WT yidi = xido * tmat[3] + yido * tmat[4] + tmat[5];

                    int outoff = setId * nimages * out.info.strides[2] +
                                 batch * out.info.strides[3];
                    int inoff = setId * nimages * in.info.strides[2] +
                                batch * in.info.strides[3];
                    const int loco =
                        outoff + (yido * out.info.strides[1] + xido);

                    constexpr int xdim = 0, ydim = 1, batch_dim = 2;
                    const bool clamp = order != 1;
                    {
                        int xid = (method == AF_INTERP_LOWER ? floor(xidi)
                                                             : round(xidi));
                        int yid = (method == AF_INTERP_LOWER ? floor(yidi)
                                                             : round(yidi));

                        const int x_lim    = in.info.dims[xdim];
                        const int y_lim    = in.info.dims[ydim];
                        const int x_stride = in.info.strides[xdim];
                        const int y_stride = in.info.strides[ydim];

                        const int idx = inoff + yid * y_stride + xid * x_stride;

                        bool condX = xid >= 0 && xid < x_lim;
                        bool condY = yid >= 0 && yid < y_lim;

                        T zero    = (T)(0);
                        bool cond = condX && condY;

                        for (int n = 0; n < limages; n++) {
                            int idx_n = idx + n * in.info.strides[batch_dim];
                            T val     = (clamp || cond) ? d_in[idx_n] : zero;
                            d_out[loco + n * out.info.strides[batch_dim]] = val;
                        }
                    }
                });
        })
        .wait();
}

}  // namespace kernel
}  // namespace oneapi
