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
#include <cudnn.hpp>
#include <err_cuda.hpp>
#include <kernel/convolve.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>
#include <af/dim4.hpp>
#include <type_traits>

using af::dim4;

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

template <typename T, typename accT>
Array<T> convolve2_cudnn(const Array<T> &signal, const Array<accT> &filter,
                         const dim4 stride, const dim4 padding,
                         const dim4 dilation) {
    auto cudnn = nnHandle();

    dim4 sDims = signal.dims();
    dim4 fDims = filter.dims();

    const int n = sDims[3];
    const int c = sDims[2];
    const int h = sDims[1];
    const int w = sDims[0];

    // create input descriptor
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    cudnnDataType_t cudnn_dtype = getCudnnDataType<accT>();
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, n, c, h, w));

    // create filter descriptor
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_descriptor, cudnn_dtype,
                                           CUDNN_TENSOR_NCHW, fDims[3],
                                           fDims[2], fDims[1], fDims[0]));

    // create convolution descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

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

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, n_out, c_out, h_out,
                                           w_out));

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

    // prepare output array and scratch space
    dim4 odims(w_out, h_out, c_out, n_out);
    Array<accT> out = createEmptyArray<accT>(odims);

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    accT alpha = scalar<accT>(1.0);
    accT beta  = scalar<accT>(0.0);
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn, &alpha, input_descriptor, cast<accT>(signal).device(),
        filter_descriptor, filter.device(), convolution_descriptor,
        convolution_algorithm, (void *)workspace_buffer.get(), workspace_bytes,
        &beta, output_descriptor, out.device()));

    // destroy all descriptors
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

    return cast<T>(out);
}

template <typename T>
constexpr void checkTypeSupport() {
    constexpr bool isValidType = (std::is_same<float, T>::value ||
                                  std::is_same<double, T>::value ||
                                  std::is_same<int, T>::value ||
                                  std::is_same<unsigned char, T>::value);
    if (!isValidType) {
        AF_ERROR("Invalid CuDNN data type:\
                  only f64, f32, s32, u8 are supported", AF_ERR_ARG);
    }
}

template <typename T, typename accT>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    checkTypeSupport<accT>();

    signal.eval();
    filter.eval();

    return convolve2_cudnn<T, accT>(signal, filter, stride, padding, dilation);
}

#define INSTANTIATE(T, accT)                                                  \
    template Array<T> convolve2<T, accT>(                                     \
        Array<T> const &signal, Array<accT> const &filter, const dim4 stride, \
        const dim4 padding, const dim4 dilation);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
#undef INSTANTIATE

template <typename T, typename accT>
Array<T> conv2FilterGradient(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<accT> &original_filter,
                             const Array<T> &convolved_output, af::dim4 stride,
                             af::dim4 padding, af::dim4 dilation) {
    auto cudnn = nnHandle();

    dim4 iDims = incoming_gradient.dims();
    dim4 sDims = original_signal.dims();
    dim4 fDims = original_filter.dims();

    // create x descriptor
    cudnnTensorDescriptor_t x_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_descriptor));
    cudnnDataType_t cudnn_dtype = getCudnnDataType<accT>();
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, sDims[3], sDims[2],
                                           sDims[1], sDims[0]));

    // create dy descriptor
    cudnnTensorDescriptor_t dy_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dy_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, iDims[3], iDims[2],
                                           iDims[1], iDims[0]));

    // create convolution descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        convolution_descriptor, padding[1], padding[0], stride[1], stride[0],
        dilation[1], dilation[0], CUDNN_CONVOLUTION, cudnn_dtype));

    // create output filter gradient descriptor
    cudnnFilterDescriptor_t dw_descriptor;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&dw_descriptor));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(dw_descriptor, cudnn_dtype,
                                           CUDNN_TENSOR_NCHW, fDims[3],
                                           fDims[2], fDims[1], fDims[0]));

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
    Array<accT> out = createEmptyArray<accT>(odims);

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    accT alpha = scalar<accT>(1.0);
    accT beta  = scalar<accT>(0.0);
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        cudnn, &alpha, x_descriptor, cast<accT>(original_signal).device(),
        dy_descriptor, cast<accT>(incoming_gradient).device(),
        convolution_descriptor, bwd_filt_convolution_algorithm,
        (void *)workspace_buffer.get(), workspace_bytes, &beta, dw_descriptor,
        out.device()));

    // destroy all descriptors
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_descriptor));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(dw_descriptor));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

    return cast<T>(out);
}

template <typename T, typename accT>
Array<T> conv2DataGradient(const Array<T> &incoming_gradient,
                           const Array<T> &original_signal,
                           const Array<accT> &original_filter,
                           const Array<T> &convolved_output, af::dim4 stride,
                           af::dim4 padding, af::dim4 dilation) {
    auto cudnn = nnHandle();

    dim4 iDims = incoming_gradient.dims();
    dim4 sDims = original_signal.dims();
    dim4 fDims = original_filter.dims();

    // create x descriptor
    cudnnTensorDescriptor_t dx_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_descriptor));
    cudnnDataType_t cudnn_dtype = getCudnnDataType<accT>();
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dx_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, sDims[3], sDims[2],
                                           sDims[1], sDims[0]));

    // create dy descriptor
    cudnnTensorDescriptor_t dy_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dy_descriptor, CUDNN_TENSOR_NCHW,
                                           cudnn_dtype, iDims[3], iDims[2],
                                           iDims[1], iDims[0]));

    // create output filter gradient descriptor
    cudnnFilterDescriptor_t w_descriptor;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_descriptor));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_descriptor, cudnn_dtype,
                                           CUDNN_TENSOR_NCHW, fDims[3],
                                           fDims[2], fDims[1], fDims[0]));

    // create convolution descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
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
    Array<accT> out = createEmptyArray<accT>(odims);

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    accT alpha = scalar<accT>(1.0);
    accT beta  = scalar<accT>(0.0);

    CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn, &alpha, w_descriptor, cast<accT>(original_filter).device(),
        dy_descriptor, cast<accT>(incoming_gradient).device(),
        convolution_descriptor, bwd_data_convolution_algorithm,
        (void *)workspace_buffer.get(), workspace_bytes, &beta, dx_descriptor,
        out.device()));

    // destroy all descriptors
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_descriptor));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_descriptor));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

    return cast<T>(out);
}

#define INSTANTIATE(T, accT)                                                  \
    template Array<T> conv2DataGradient<T, accT>(                             \
        Array<T> const &incoming_gradient, Array<T> const &original_signal,   \
        Array<accT> const &original_filter, Array<T> const &convolved_output, \
        const dim4 stride, const dim4 padding, const dim4 dilation);          \
    template Array<T> conv2FilterGradient<T, accT>(                           \
        Array<T> const &incoming_gradient, Array<T> const &original_signal,   \
        Array<accT> const &original_filter, Array<T> const &convolved_output, \
        const dim4 stride, const dim4 padding, const dim4 dilation);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
#undef INSTANTIATE
}
