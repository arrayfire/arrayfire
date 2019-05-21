/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <cast.hpp>
#include <convolve.hpp>
#include <common/unique_handle.hpp>
#include <cudnn.hpp>
#include <err_cuda.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>
#include <type_traits>

using af::dim4;
using common::make_handle;
using common::unique_handle;

namespace cuda {

template<typename T> cudnnDataType_t getCudnnDataType() {
    AF_ERROR("Invalid CuDNN data type", AF_ERR_ARG);
    return CUDNN_DATA_FLOAT;
}
template<> cudnnDataType_t getCudnnDataType<float>() {  return CUDNN_DATA_FLOAT; }
template<> cudnnDataType_t getCudnnDataType<double>() {  return CUDNN_DATA_DOUBLE; }
template<> cudnnDataType_t getCudnnDataType<int>() {  return CUDNN_DATA_INT32; }
template<> cudnnDataType_t getCudnnDataType<unsigned char>() {  return CUDNN_DATA_UINT8; }

template <typename T, typename accT, dim_t baseDim, bool expand>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  AF_BATCH_KIND kind) {
    const dim4 sDims = signal.dims();
    const dim4 fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (dim_t d = 0; d < 4; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < baseDim ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
            for (dim_t i = baseDim; i < 4; ++i) oDims[i] = fDims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::convolve_nd<T, accT>(out, signal, filter, kind, baseDim, expand);

    return out;
}

void cudnnSet(cudnnTensorDescriptor_t desc, cudnnDataType_t cudnn_dtype, dim4 dims) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, dims[3], dims[2],
                                           dims[1], dims[0]));
}

void cudnnSet(cudnnFilterDescriptor_t desc, cudnnDataType_t cudnn_dtype, dim4 dims) {
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc, cudnn_dtype,
                                           CUDNN_TENSOR_NCHW, dims[3],
                                           dims[2], dims[1], dims[0]));
}

template<typename Desc, typename T>
unique_handle<Desc> toCudnn(Array<T> arr) {
    dim4 dims = arr.dims();

    auto descriptor = make_handle<Desc>();
    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();
    cudnnSet(descriptor, cudnn_dtype, dims);
    return descriptor;
}

template <typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &c_filter,
                   Array<accT> const &r_filter) {
    const dim4 cfDims = c_filter.dims();
    const dim4 rfDims = r_filter.dims();

    const dim_t cfLen = cfDims.elements();
    const dim_t rfLen = rfDims.elements();

    const dim4 sDims = signal.dims();
    dim4 tDims       = sDims;
    dim4 oDims       = sDims;

    if (expand) {
        tDims[0] += cfLen - 1;
        oDims[0] += cfLen - 1;
        oDims[1] += rfLen - 1;
    }

    Array<T> temp = createEmptyArray<T>(tDims);
    Array<T> out  = createEmptyArray<T>(oDims);

    kernel::convolve2<T, accT>(temp, signal, c_filter, 0, expand);
    kernel::convolve2<T, accT>(out, temp, r_filter, 1, expand);

    return out;
}

#define INSTANTIATE(T, accT)                                                 \
    template Array<T> convolve<T, accT, 1, true>(Array<T> const &signal,     \
                                                 Array<accT> const &filter,  \
                                                 AF_BATCH_KIND kind);        \
    template Array<T> convolve<T, accT, 1, false>(Array<T> const &signal,    \
                                                  Array<accT> const &filter, \
                                                  AF_BATCH_KIND kind);       \
    template Array<T> convolve<T, accT, 2, true>(Array<T> const &signal,     \
                                                 Array<accT> const &filter,  \
                                                 AF_BATCH_KIND kind);        \
    template Array<T> convolve<T, accT, 2, false>(Array<T> const &signal,    \
                                                  Array<accT> const &filter, \
                                                  AF_BATCH_KIND kind);       \
    template Array<T> convolve<T, accT, 3, true>(Array<T> const &signal,     \
                                                 Array<accT> const &filter,  \
                                                 AF_BATCH_KIND kind);        \
    template Array<T> convolve<T, accT, 3, false>(Array<T> const &signal,    \
                                                  Array<accT> const &filter, \
                                                  AF_BATCH_KIND kind);       \
    template Array<T> convolve2<T, accT, true>(Array<T> const &signal,       \
                                               Array<accT> const &c_filter,  \
                                               Array<accT> const &r_filter); \
    template Array<T> convolve2<T, accT, false>(Array<T> const &signal,      \
                                                Array<accT> const &c_filter, \
                                                Array<accT> const &r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)
#undef INSTANTIATE

template <typename T>
Array<T> convolve2_cudnn(const Array<T> &signal, const Array<T> &filter,
                         const dim4 stride, const dim4 padding,
                         const dim4 dilation) {
    auto cudnn = nnHandle();

    dim4 sDims = signal.dims();
    dim4 fDims = filter.dims();

    const int n = sDims[3];
    const int c = sDims[2];
    const int h = sDims[1];
    const int w = sDims[0];

    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();
    auto input_descriptor = toCudnn<cudnnTensorDescriptor_t>(signal);
    auto filter_descriptor = toCudnn<cudnnFilterDescriptor_t>(filter);

    // create convolution descriptor
    auto convolution_descriptor = make_handle<cudnnConvolutionDescriptor_t>();
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        convolution_descriptor, padding[1], padding[0], stride[1], stride[0],
        dilation[1], dilation[0], CUDNN_CONVOLUTION, cudnn_dtype));

    // get output dimensions
    const int tensorDims = 4;
    int convolved_output_dim[tensorDims];
    CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
        convolution_descriptor, input_descriptor, filter_descriptor, tensorDims,
        convolved_output_dim));

    // create output descriptor
    const int n_out = convolved_output_dim[0];
    const int c_out = convolved_output_dim[1];
    const int h_out = convolved_output_dim[2];
    const int w_out = convolved_output_dim[3];

    // prepare output array and scratch space
    dim4 odims(w_out, h_out, c_out, n_out);
    Array<T> out = createEmptyArray<T>(odims);

    auto output_descriptor = toCudnn<cudnnTensorDescriptor_t>(out);

    // get convolution algorithm
    const int memory_limit =
        0;  // TODO: set to remaining space in memory manager?
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        cudnn, input_descriptor, filter_descriptor, convolution_descriptor,
        output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, memory_limit,
        &convolution_algorithm));

    // figure out scratch space memory requirements
    size_t workspace_bytes;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, input_descriptor, filter_descriptor, convolution_descriptor,
        output_descriptor, convolution_algorithm, &workspace_bytes));

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    T alpha = scalar<T>(1.0);
    T beta  = scalar<T>(0.0);
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn, &alpha, input_descriptor, signal.device(),
        filter_descriptor, filter.device(), convolution_descriptor,
        convolution_algorithm, (void *)workspace_buffer.get(), workspace_bytes,
        &beta, output_descriptor, out.device()));

    return out;
}

template <typename T>
constexpr void checkTypeSupport() {
    static_assert (std::is_same<float, T>::value ||
                   std::is_same<double, T>::value,
      "Invalid CuDNN data type: only f64, f32 are supported");
}

template <typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    checkTypeSupport<T>();
    return convolve2_cudnn<T>(signal, filter, stride, padding, dilation);
}

#define INSTANTIATE(T)                                                  \
    template Array<T> convolve2<T>(                                     \
        Array<T> const &signal, Array<T> const &filter, const dim4 stride, \
        const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
#undef INSTANTIATE

template <typename T>
Array<T> conv2FilterGradient(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<T> &original_filter,
                             const Array<T> &convolved_output, af::dim4 stride,
                             af::dim4 padding, af::dim4 dilation) {
    auto cudnn = nnHandle();

    dim4 iDims = incoming_gradient.dims();
    dim4 sDims = original_signal.dims();
    dim4 fDims = original_filter.dims();

    // create dx descriptor
    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();
    auto x_descriptor = toCudnn<cudnnTensorDescriptor_t>(original_signal);
    auto dy_descriptor = toCudnn<cudnnTensorDescriptor_t>(incoming_gradient);

    // create convolution descriptor
    auto convolution_descriptor = make_handle<cudnnConvolutionDescriptor_t>();
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        convolution_descriptor, padding[1], padding[0], stride[1], stride[0],
        dilation[1], dilation[0], CUDNN_CONVOLUTION, cudnn_dtype));

    // create output filter gradient descriptor
    auto dw_descriptor = toCudnn<cudnnFilterDescriptor_t>(original_filter);

    // determine algorithm to use
    cudnnConvolutionBwdFilterAlgo_t bwd_filt_convolution_algorithm;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn, x_descriptor, dy_descriptor, convolution_descriptor,
        dw_descriptor, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
        &bwd_filt_convolution_algorithm));

    // figure out scratch space memory requirements
    size_t workspace_bytes;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn, x_descriptor, dy_descriptor, convolution_descriptor,
        dw_descriptor, bwd_filt_convolution_algorithm, &workspace_bytes));
    // prepare output array and scratch space
    dim4 odims(fDims[0], fDims[1], fDims[2], fDims[3]);
    Array<T> out = createEmptyArray<T>(odims);

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    T alpha = scalar<T>(1.0);
    T beta  = scalar<T>(0.0);
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        cudnn, &alpha, x_descriptor, original_signal.device(),
        dy_descriptor, incoming_gradient.device(),
        convolution_descriptor, bwd_filt_convolution_algorithm,
        (void *)workspace_buffer.get(), workspace_bytes, &beta, dw_descriptor,
        out.device()));

    return out;
}



template <typename T>
Array<T> conv2DataGradient(const Array<T> &incoming_gradient,
                           const Array<T> &original_signal,
                           const Array<T> &original_filter,
                           const Array<T> &convolved_output, af::dim4 stride,
                           af::dim4 padding, af::dim4 dilation) {
    auto cudnn = nnHandle();

    dim4 iDims = incoming_gradient.dims();
    dim4 sDims = original_signal.dims();
    dim4 fDims = original_filter.dims();

    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();

    // create x descriptor
    auto dx_descriptor = toCudnn<cudnnTensorDescriptor_t>(original_signal);
    auto dy_descriptor = toCudnn<cudnnTensorDescriptor_t>(incoming_gradient);

    // create output filter gradient descriptor
    auto w_descriptor = make_handle<cudnnFilterDescriptor_t>();
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_descriptor, cudnn_dtype,
                                           CUDNN_TENSOR_NCHW, fDims[3],
                                           fDims[2], fDims[1], fDims[0]));

    // create convolution descriptor
    auto convolution_descriptor = make_handle<cudnnConvolutionDescriptor_t>();
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        convolution_descriptor, padding[1], padding[0], stride[1], stride[0],
        dilation[1], dilation[0], CUDNN_CONVOLUTION, cudnn_dtype));

    cudnnConvolutionBwdDataAlgo_t bwd_data_convolution_algorithm;
    if (dilation[0] == 1 && dilation[1] == 1) {
        bwd_data_convolution_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else {
        bwd_data_convolution_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    }

    // figure out scratch space memory requirements
    size_t workspace_bytes;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn, w_descriptor, dy_descriptor, convolution_descriptor,
        dx_descriptor, bwd_data_convolution_algorithm, &workspace_bytes));

    dim4 odims(sDims[0], sDims[1], sDims[2], sDims[3]);
    Array<T> out = createEmptyArray<T>(odims);

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    T alpha = scalar<T>(1.0);
    T beta  = scalar<T>(0.0);

    CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn, &alpha, w_descriptor,
        original_filter.get(), dy_descriptor, incoming_gradient.get(),
        convolution_descriptor, bwd_data_convolution_algorithm,
        (void *)workspace_buffer.get(), workspace_bytes, &beta, dx_descriptor,
        out.device()));

    return out;
}

#define INSTANTIATE(T)                                                  \
    template Array<T> conv2DataGradient<T>(                          \
        Array<T> const &incoming_gradient, Array<T> const &original_signal,   \
        Array<T> const &original_filter, Array<T> const &convolved_output, \
        const dim4 stride, const dim4 padding, const dim4 dilation);          \
    template Array<T> conv2FilterGradient<T>(                        \
        Array<T> const &incoming_gradient, Array<T> const &original_signal,   \
        Array<T> const &original_filter, Array<T> const &convolved_output, \
        const dim4 stride, const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
#undef INSTANTIATE

} //cuda namespace
