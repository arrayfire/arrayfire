/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/constants.h>
#include <program.hpp>
#include <common/dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <kernel/convolve_separable.hpp>
#include <kernel/gradient.hpp>
#include <kernel/sort_by_key.hpp>
#include <kernel/range.hpp>
#include <kernel_headers/harris.hpp>
#include <memory.hpp>
#include <cache.hpp>
#include <tuple>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;

namespace opencl
{
namespace kernel
{
static const unsigned HARRIS_THREADS_PER_GROUP = 256;
static const unsigned HARRIS_THREADS_X = 16;
static const unsigned HARRIS_THREADS_Y = HARRIS_THREADS_PER_GROUP / HARRIS_THREADS_X;

template<typename T>
void gaussian1D(T* out, const int dim, double sigma=0.0)
{
    if(!(sigma>0)) sigma = 0.25*dim;

    T sum = (T)0;
    for(int i=0;i<dim;i++)
    {
        int x = i-(dim-1)/2;
        T el = 1. / sqrt(2 * af::Pi * sigma*sigma) * exp(-((x*x)/(2*(sigma*sigma))));
        out[i] = el;
        sum   += el;
    }

    for(int k=0;k<dim;k++)
        out[k] /= sum;
}

template<typename T, typename convAccT>
void conv_helper(Param &ixx, Param &ixy, Param &iyy, Param &filter)
{
    Param ixx_tmp, ixy_tmp, iyy_tmp;
    ixx_tmp.info.offset = ixy_tmp.info.offset = iyy_tmp.info.offset = 0;
    for (dim_t i = 0; i < 4; i++) {
        ixx_tmp.info.dims[i] = ixx.info.dims[i];
        ixy_tmp.info.dims[i] = ixy.info.dims[i];
        iyy_tmp.info.dims[i] = iyy.info.dims[i];
        ixx_tmp.info.strides[i] = ixx.info.strides[i];
        ixy_tmp.info.strides[i] = ixy.info.strides[i];
        iyy_tmp.info.strides[i] = iyy.info.strides[i];
    }
    ixx_tmp.data = bufferAlloc(ixx_tmp.info.dims[3] * ixx_tmp.info.strides[3] * sizeof(convAccT));
    ixy_tmp.data = bufferAlloc(ixy_tmp.info.dims[3] * ixy_tmp.info.strides[3] * sizeof(convAccT));
    iyy_tmp.data = bufferAlloc(iyy_tmp.info.dims[3] * iyy_tmp.info.strides[3] * sizeof(convAccT));

    convSep<T, convAccT, 0, false>(ixx_tmp, ixx, filter);
    convSep<T, convAccT, 1, false>(ixx, ixx_tmp, filter);
    convSep<T, convAccT, 0, false>(ixy_tmp, ixy, filter);
    convSep<T, convAccT, 1, false>(ixy, ixy_tmp, filter);
    convSep<T, convAccT, 0, false>(iyy_tmp, iyy, filter);
    convSep<T, convAccT, 1, false>(iyy, iyy_tmp, filter);

    bufferFree(ixx_tmp.data);
    bufferFree(ixy_tmp.data);
    bufferFree(iyy_tmp.data);
}

template<typename T>
std::tuple<cl::Kernel*, cl::Kernel*, cl::Kernel*, cl::Kernel*>
getHarrisKernels()
{
    static const char* kernelNames[4] =
        {"second_order_deriv", "keep_corners", "harris_responses", "non_maximal"};

    kc_entry_t entries[4];

    int device = getActiveDeviceId();

    std::string checkName = kernelNames[0] + std::string("_") + std::string(dtype_traits<T>::getName());

    entries[0] = kernelCache(device, checkName);

    if (entries[0].prog==0 && entries[0].ker==0)
    {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {harris_cl};
        const int   ker_lens[] = {harris_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());

        for (int i=0; i<4; ++i)
        {
            entries[i].prog = new Program(prog);
            entries[i].ker  = new Kernel(*entries[i].prog, kernelNames[i]);

            std::string name = kernelNames[i] +
                std::string("_") + std::string(dtype_traits<T>::getName());

            addKernelToCache(device, name, entries[i]);
        }
    } else {
        for (int i=1; i<4; ++i) {
            std::string name = kernelNames[i] +
                std::string("_") + std::string(dtype_traits<T>::getName());

            entries[i] = kernelCache(device, name);
        }
    }

    return std::make_tuple(entries[0].ker, entries[1].ker, entries[2].ker, entries[3].ker);
}

template<typename T, typename convAccT>
void harris(unsigned* corners_out,
            Param &x_out,
            Param &y_out,
            Param &resp_out,
            Param in,
            const unsigned max_corners,
            const float min_response,
            const float sigma,
            const unsigned filter_len,
            const float k_thr)
{
    auto kernels = getHarrisKernels<T>();

    // Window filter
    convAccT* h_filter = new convAccT[filter_len];
    // Decide between rectangular or circular filter
    if (sigma < 0.5f) {
        for (unsigned i = 0; i < filter_len; i++)
            h_filter[i] = (T)1.f / (filter_len);
    } else {
        gaussian1D<convAccT>(h_filter, (int)filter_len, sigma);
    }

    const unsigned border_len = filter_len / 2 + 1;

    // Copy filter to device object
    Param filter;
    filter.info.dims[0] = filter_len;
    filter.info.strides[0] = 1;
    filter.info.offset = 0;

    for (int k = 1; k < 4; k++) {
        filter.info.dims[k] = 1;
        filter.info.strides[k] = filter.info.dims[k - 1] * filter.info.strides[k - 1];
    }

    int filter_elem = filter.info.strides[3] * filter.info.dims[3];
    filter.data = bufferAlloc(filter_elem * sizeof(convAccT));
    getQueue().enqueueWriteBuffer(*filter.data, CL_TRUE, 0, filter_elem * sizeof(convAccT), h_filter);

    Param ix, iy;
    ix.info.offset = iy.info.offset = 0;
    for (dim_t i = 0; i < 4; i++) {
        ix.info.dims[i] = iy.info.dims[i] = in.info.dims[i];
        ix.info.strides[i] = iy.info.strides[i] = in.info.strides[i];
    }
    ix.data = bufferAlloc(ix.info.dims[3] * ix.info.strides[3] * sizeof(T));
    iy.data = bufferAlloc(iy.info.dims[3] * iy.info.strides[3] * sizeof(T));

    // Compute first-order derivatives as gradients
    gradient<T>(iy, ix, in);

    Param ixx, ixy, iyy;
    ixx.info.offset = ixy.info.offset = iyy.info.offset = 0;
    for (dim_t i = 0; i < 4; i++) {
        ixx.info.dims[i] = ixy.info.dims[i] = iyy.info.dims[i] = in.info.dims[i];
        ixx.info.strides[i] = ixy.info.strides[i] = iyy.info.strides[i] = in.info.strides[i];
    }
    ixx.data = bufferAlloc(ixx.info.dims[3] * ixx.info.strides[3] * sizeof(T));
    ixy.data = bufferAlloc(ixy.info.dims[3] * ixy.info.strides[3] * sizeof(T));
    iyy.data = bufferAlloc(iyy.info.dims[3] * iyy.info.strides[3] * sizeof(T));

    // Second order-derivatives kernel sizes
    const unsigned blk_x_so = divup(in.info.dims[3] * in.info.strides[3], HARRIS_THREADS_PER_GROUP);
    const NDRange local_so(HARRIS_THREADS_PER_GROUP, 1);
    const NDRange global_so(blk_x_so * HARRIS_THREADS_PER_GROUP, 1);

    auto soOp = KernelFunctor< Buffer, Buffer, Buffer,
                               unsigned, Buffer, Buffer > (*std::get<0>(kernels));

    // Compute second-order derivatives
    soOp(EnqueueArgs(getQueue(), global_so, local_so),
         *ixx.data, *ixy.data, *iyy.data,
         in.info.dims[3] * in.info.strides[3], *ix.data, *iy.data);
    CL_DEBUG_FINISH(getQueue());

    bufferFree(ix.data);
    bufferFree(iy.data);

    // Convolve second order derivatives with proper window filter
    conv_helper<T, convAccT>(ixx, ixy, iyy, filter);
    bufferFree(filter.data);

    cl::Buffer *d_responses = bufferAlloc(in.info.dims[3] * in.info.strides[3] * sizeof(T));

    // Harris responses kernel sizes
    unsigned blk_x_hr = divup(in.info.dims[0] - border_len*2, HARRIS_THREADS_X);
    unsigned blk_y_hr = divup(in.info.dims[1] - border_len*2, HARRIS_THREADS_Y);
    const NDRange local_hr(HARRIS_THREADS_X, HARRIS_THREADS_Y);
    const NDRange global_hr(blk_x_hr * HARRIS_THREADS_X, blk_y_hr * HARRIS_THREADS_Y);

    auto hrOp = KernelFunctor< Buffer, unsigned, unsigned, Buffer, Buffer, Buffer,
                               float, unsigned> (*std::get<2>(kernels));

    // Calculate Harris responses for all pixels
    hrOp(EnqueueArgs(getQueue(), global_hr, local_hr),
         *d_responses, in.info.dims[0], in.info.dims[1],
         *ixx.data, *ixy.data, *iyy.data, k_thr, border_len);
    CL_DEBUG_FINISH(getQueue());

    bufferFree(ixx.data);
    bufferFree(ixy.data);
    bufferFree(iyy.data);

    // Number of corners is not known a priori, limit maximum number of corners
    // according to image dimensions
    unsigned corner_lim = in.info.dims[3] * in.info.strides[3] * 0.2f;

    unsigned corners_found = 0;
    cl::Buffer *d_corners_found = bufferAlloc(sizeof(unsigned));
    getQueue().enqueueWriteBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned), &corners_found);

    cl::Buffer *d_x_corners = bufferAlloc(corner_lim * sizeof(float));
    cl::Buffer *d_y_corners = bufferAlloc(corner_lim * sizeof(float));
    cl::Buffer *d_resp_corners = bufferAlloc(corner_lim * sizeof(float));

    const float min_r = (max_corners > 0) ? 0.f : min_response;

    auto nmOp = KernelFunctor< Buffer, Buffer, Buffer, Buffer, Buffer, unsigned, unsigned,
                            float, unsigned, unsigned> (*std::get<3>(kernels));

    // Perform non-maximal suppression
    nmOp(EnqueueArgs(getQueue(), global_hr, local_hr),
         *d_x_corners, *d_y_corners, *d_resp_corners, *d_corners_found,
         *d_responses, in.info.dims[0], in.info.dims[1],
         min_r, border_len, corner_lim);
    CL_DEBUG_FINISH(getQueue());

    getQueue().enqueueReadBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned), &corners_found);

    bufferFree(d_responses);
    bufferFree(d_corners_found);

    *corners_out = min(corners_found, (max_corners > 0) ? max_corners : corner_lim);
    if (*corners_out == 0) return;

    // Set output Param info
    x_out.info.dims[0] = y_out.info.dims[0] = resp_out.info.dims[0] = *corners_out;
    x_out.info.strides[0] = y_out.info.strides[0] = resp_out.info.strides[0] = 1;
    x_out.info.offset = y_out.info.offset = resp_out.info.offset = 0;
    for (int k = 1; k < 4; k++) {
        x_out.info.dims[k] = y_out.info.dims[k] = resp_out.info.dims[k] =  1;
        x_out.info.strides[k] = x_out.info.dims[k - 1] * x_out.info.strides[k - 1];
        y_out.info.strides[k] = y_out.info.dims[k - 1] * y_out.info.strides[k - 1];
        resp_out.info.strides[k] = resp_out.info.dims[k - 1] * resp_out.info.strides[k - 1];
    }

    if (max_corners > 0 && corners_found > *corners_out) {
        Param harris_resp;
        Param harris_idx;

        harris_resp.info.dims[0] = harris_idx.info.dims[0] = corners_found;
        harris_resp.info.strides[0] = harris_idx.info.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            harris_resp.info.dims[k] = 1;
            harris_resp.info.strides[k] = harris_resp.info.dims[k - 1] * harris_resp.info.strides[k - 1];
            harris_idx.info.dims[k] = 1;
            harris_idx.info.strides[k] = harris_idx.info.dims[k - 1] * harris_idx.info.strides[k - 1];
        }

        int sort_elem = harris_resp.info.strides[3] * harris_resp.info.dims[3];
        harris_resp.data = d_resp_corners;
        // Create indices using range
        harris_idx.data = bufferAlloc(sort_elem * sizeof(unsigned));
        kernel::range<uint>(harris_idx, 0);

        // Sort Harris responses
        kernel::sort0ByKey<float, uint>(harris_resp, harris_idx, false);

        x_out.data = bufferAlloc(*corners_out * sizeof(float));
        y_out.data = bufferAlloc(*corners_out * sizeof(float));
        resp_out.data = bufferAlloc(*corners_out * sizeof(float));

        // Keep corners kernel sizes
        const unsigned blk_x_kc = divup(*corners_out, HARRIS_THREADS_PER_GROUP);
        const NDRange local_kc(HARRIS_THREADS_PER_GROUP, 1);
        const NDRange global_kc(blk_x_kc * HARRIS_THREADS_PER_GROUP, 1);

        auto kcOp = KernelFunctor< Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer,
                                   unsigned> (*std::get<1>(kernels));

        // Keep only the first corners_to_keep corners with higher Harris
        // responses
        kcOp(EnqueueArgs(getQueue(), global_kc, local_kc),
             *x_out.data, *y_out.data, *resp_out.data,
             *d_x_corners, *d_y_corners, *harris_resp.data, *harris_idx.data,
             *corners_out);
        CL_DEBUG_FINISH(getQueue());

        bufferFree(d_x_corners);
        bufferFree(d_y_corners);
        bufferFree(harris_resp.data);
        bufferFree(harris_idx.data);
    }
    else if (max_corners == 0 && corners_found < corner_lim) {
        x_out.data = bufferAlloc(*corners_out * sizeof(float));
        y_out.data = bufferAlloc(*corners_out * sizeof(float));
        resp_out.data = bufferAlloc(*corners_out * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_x_corners, *x_out.data, 0, 0, *corners_out * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_y_corners, *y_out.data, 0, 0, *corners_out * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_resp_corners, *resp_out.data, 0, 0, *corners_out * sizeof(float));

        bufferFree(d_x_corners);
        bufferFree(d_y_corners);
        bufferFree(d_resp_corners);
    }
    else {
        x_out.data = d_x_corners;
        y_out.data = d_y_corners;
        resp_out.data = d_resp_corners;
    }
}
} //namespace kernel
} //namespace opencl
