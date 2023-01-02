/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <debug_opencl.hpp>
#include <kernel/convolve_separable.hpp>
#include <kernel/gradient.hpp>
#include <kernel/range.hpp>
#include <kernel/sort_by_key.hpp>
#include <kernel_headers/harris.hpp>
#include <memory.hpp>
#include <af/constants.h>
#include <af/defines.h>

#include <array>
#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void gaussian1D(T *out, const int dim, double sigma = 0.0) {
    if (!(sigma > 0)) sigma = 0.25 * dim;

    T sum = (T)0;
    for (int i = 0; i < dim; i++) {
        int x = i - (dim - 1) / 2;
        T el  = 1. / sqrt(2 * af::Pi * sigma * sigma) *
               exp(-((x * x) / (2 * (sigma * sigma))));
        out[i] = el;
        sum += el;
    }

    for (int k = 0; k < dim; k++) out[k] /= sum;
}

template<typename T, typename convAccT>
void conv_helper(Array<T> &ixx, Array<T> &ixy, Array<T> &iyy,
                 Array<convAccT> &filter) {
    Array<convAccT> ixx_tmp = createEmptyArray<convAccT>(ixx.dims());
    Array<convAccT> ixy_tmp = createEmptyArray<convAccT>(ixy.dims());
    Array<convAccT> iyy_tmp = createEmptyArray<convAccT>(iyy.dims());

    convSep<T, convAccT>(ixx_tmp, ixx, filter, 0, false);
    convSep<T, convAccT>(ixx, ixx_tmp, filter, 1, false);
    convSep<T, convAccT>(ixy_tmp, ixy, filter, 0, false);
    convSep<T, convAccT>(ixy, ixy_tmp, filter, 1, false);
    convSep<T, convAccT>(iyy_tmp, iyy, filter, 0, false);
    convSep<T, convAccT>(iyy, iyy_tmp, filter, 1, false);
}

template<typename T>
std::array<Kernel, 4> getHarrisKernels() {
    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    return {
        common::getKernel("second_order_deriv", {{harris_cl_src}}, targs,
                          options),
        common::getKernel("keep_corners", {{harris_cl_src}}, targs, options),
        common::getKernel("harris_responses", {{harris_cl_src}}, targs,
                          options),
        common::getKernel("non_maximal", {{harris_cl_src}}, targs, options),
    };
}

template<typename T, typename convAccT>
void harris(unsigned *corners_out, Param &x_out, Param &y_out, Param &resp_out,
            Param in, const unsigned max_corners, const float min_response,
            const float sigma, const unsigned filter_len, const float k_thr) {
    constexpr unsigned HARRIS_THREADS_PER_GROUP = 256;
    constexpr unsigned HARRIS_THREADS_X         = 16;
    constexpr unsigned HARRIS_THREADS_Y =
        HARRIS_THREADS_PER_GROUP / HARRIS_THREADS_X;

    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto kernels = getHarrisKernels<T>();
    auto soOp    = kernels[0];
    auto kcOp    = kernels[1];
    auto hrOp    = kernels[2];
    auto nmOp    = kernels[3];

    // Window filter
    std::vector<convAccT> h_filter(filter_len);
    // Decide between rectangular or circular filter
    if (sigma < 0.5f) {
        for (unsigned i = 0; i < filter_len; i++)
            h_filter[i] = (T)1.f / (filter_len);
    } else {
        gaussian1D<convAccT>(h_filter.data(), (int)filter_len, sigma);
    }

    const unsigned border_len = filter_len / 2 + 1;

    // Copy filter to device object
    Array<convAccT> filter =
        createHostDataArray<convAccT>(filter_len, h_filter.data());
    Array<T> ix = createEmptyArray<T>(dim4(4, in.info.dims));
    Array<T> iy = createEmptyArray<T>(dim4(4, in.info.dims));

    // Compute first-order derivatives as gradients
    gradient<T>(iy, ix, in);

    Array<T> ixx = createEmptyArray<T>(dim4(4, in.info.dims));
    Array<T> ixy = createEmptyArray<T>(dim4(4, in.info.dims));
    Array<T> iyy = createEmptyArray<T>(dim4(4, in.info.dims));

    // Second order-derivatives kernel sizes
    const unsigned blk_x_so =
        divup(in.info.dims[3] * in.info.strides[3], HARRIS_THREADS_PER_GROUP);
    const NDRange local_so(HARRIS_THREADS_PER_GROUP, 1);
    const NDRange global_so(blk_x_so * HARRIS_THREADS_PER_GROUP, 1);

    // Compute second-order derivatives
    soOp(EnqueueArgs(getQueue(), global_so, local_so), *ixx.get(), *ixy.get(),
         *iyy.get(), in.info.dims[3] * in.info.strides[3], *ix.get(),
         *iy.get());
    CL_DEBUG_FINISH(getQueue());

    // Convolve second order derivatives with proper window filter
    conv_helper<T, convAccT>(ixx, ixy, iyy, filter);

    cl::Buffer *d_responses =
        bufferAlloc(in.info.dims[3] * in.info.strides[3] * sizeof(T));

    // Harris responses kernel sizes
    unsigned blk_x_hr =
        divup(in.info.dims[0] - border_len * 2, HARRIS_THREADS_X);
    unsigned blk_y_hr =
        divup(in.info.dims[1] - border_len * 2, HARRIS_THREADS_Y);
    const NDRange local_hr(HARRIS_THREADS_X, HARRIS_THREADS_Y);
    const NDRange global_hr(blk_x_hr * HARRIS_THREADS_X,
                            blk_y_hr * HARRIS_THREADS_Y);

    // Calculate Harris responses for all pixels
    hrOp(EnqueueArgs(getQueue(), global_hr, local_hr), *d_responses,
         static_cast<uint>(in.info.dims[0]), static_cast<uint>(in.info.dims[1]),
         *ixx.get(), *ixy.get(), *iyy.get(), k_thr, border_len);
    CL_DEBUG_FINISH(getQueue());

    // Number of corners is not known a priori, limit maximum number of corners
    // according to image dimensions
    unsigned corner_lim = in.info.dims[3] * in.info.strides[3] * 0.2f;

    unsigned corners_found      = 0;
    cl::Buffer *d_corners_found = bufferAlloc(sizeof(unsigned));
    getQueue().enqueueFillBuffer(*d_corners_found, corners_found, 0,
                                 sizeof(unsigned));

    cl::Buffer *d_x_corners    = bufferAlloc(corner_lim * sizeof(float));
    cl::Buffer *d_y_corners    = bufferAlloc(corner_lim * sizeof(float));
    cl::Buffer *d_resp_corners = bufferAlloc(corner_lim * sizeof(float));

    const float min_r = (max_corners > 0) ? 0.f : min_response;

    // Perform non-maximal suppression
    nmOp(EnqueueArgs(getQueue(), global_hr, local_hr), *d_x_corners,
         *d_y_corners, *d_resp_corners, *d_corners_found, *d_responses,
         static_cast<uint>(in.info.dims[0]), static_cast<uint>(in.info.dims[1]),
         min_r, border_len, corner_lim);
    CL_DEBUG_FINISH(getQueue());

    getQueue().enqueueReadBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned),
                                 &corners_found);

    bufferFree(d_responses);
    bufferFree(d_corners_found);

    *corners_out =
        min(corners_found, (max_corners > 0) ? max_corners : corner_lim);
    if (*corners_out == 0) return;

    // Set output Param info
    x_out.info.dims[0] = y_out.info.dims[0] = resp_out.info.dims[0] =
        *corners_out;
    x_out.info.strides[0] = y_out.info.strides[0] = resp_out.info.strides[0] =
        1;
    x_out.info.offset = y_out.info.offset = resp_out.info.offset = 0;
    for (int k = 1; k < 4; k++) {
        x_out.info.dims[k] = y_out.info.dims[k] = resp_out.info.dims[k] = 1;
        x_out.info.strides[k] =
            x_out.info.dims[k - 1] * x_out.info.strides[k - 1];
        y_out.info.strides[k] =
            y_out.info.dims[k - 1] * y_out.info.strides[k - 1];
        resp_out.info.strides[k] =
            resp_out.info.dims[k - 1] * resp_out.info.strides[k - 1];
    }

    if (max_corners > 0 && corners_found > *corners_out) {
        Param harris_resp;
        Param harris_idx;

        harris_resp.info.dims[0] = harris_idx.info.dims[0] = corners_found;
        harris_resp.info.strides[0] = harris_idx.info.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            harris_resp.info.dims[k] = 1;
            harris_resp.info.strides[k] =
                harris_resp.info.dims[k - 1] * harris_resp.info.strides[k - 1];
            harris_idx.info.dims[k] = 1;
            harris_idx.info.strides[k] =
                harris_idx.info.dims[k - 1] * harris_idx.info.strides[k - 1];
        }

        int sort_elem = harris_resp.info.strides[3] * harris_resp.info.dims[3];
        harris_resp.data = d_resp_corners;
        // Create indices using range
        harris_idx.data = bufferAlloc(sort_elem * sizeof(unsigned));
        kernel::range<uint>(harris_idx, 0);

        // Sort Harris responses
        kernel::sort0ByKey<float, uint>(harris_resp, harris_idx, false);

        x_out.data    = bufferAlloc(*corners_out * sizeof(float));
        y_out.data    = bufferAlloc(*corners_out * sizeof(float));
        resp_out.data = bufferAlloc(*corners_out * sizeof(float));

        // Keep corners kernel sizes
        const unsigned blk_x_kc = divup(*corners_out, HARRIS_THREADS_PER_GROUP);
        const NDRange local_kc(HARRIS_THREADS_PER_GROUP, 1);
        const NDRange global_kc(blk_x_kc * HARRIS_THREADS_PER_GROUP, 1);

        // Keep only the first corners_to_keep corners with higher Harris
        // responses
        kcOp(EnqueueArgs(getQueue(), global_kc, local_kc), *x_out.data,
             *y_out.data, *resp_out.data, *d_x_corners, *d_y_corners,
             *harris_resp.data, *harris_idx.data, *corners_out);
        CL_DEBUG_FINISH(getQueue());

        bufferFree(d_x_corners);
        bufferFree(d_y_corners);
        bufferFree(harris_resp.data);
        bufferFree(harris_idx.data);
    } else if (max_corners == 0 && corners_found < corner_lim) {
        x_out.data    = bufferAlloc(*corners_out * sizeof(float));
        y_out.data    = bufferAlloc(*corners_out * sizeof(float));
        resp_out.data = bufferAlloc(*corners_out * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_x_corners, *x_out.data, 0, 0,
                                     *corners_out * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_y_corners, *y_out.data, 0, 0,
                                     *corners_out * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_resp_corners, *resp_out.data, 0, 0,
                                     *corners_out * sizeof(float));

        bufferFree(d_x_corners);
        bufferFree(d_y_corners);
        bufferFree(d_resp_corners);
    } else {
        x_out.data    = d_x_corners;
        y_out.data    = d_y_corners;
        resp_out.data = d_resp_corners;
    }
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
