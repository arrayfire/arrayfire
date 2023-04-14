/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <convolve.hpp>

#include <Array.hpp>
#include <blas.hpp>
#include <common/cast.hpp>
#include <common/half.hpp>
#include <common/indexing_helpers.hpp>
#include <common/moddims.hpp>
#include <common/unique_handle.hpp>
#ifdef WITH_CUDNN
#include <cudnn.hpp>
#endif
#include <err_cuda.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <reorder.hpp>
#include <transpose.hpp>
#include <unwrap.hpp>
#include <wrap.hpp>
#include <af/dim4.hpp>

#include <type_traits>
#include <utility>
#include <vector>

using af::dim4;
using arrayfire::common::flip;
using arrayfire::common::half;
using arrayfire::common::make_handle;
using arrayfire::common::modDims;
using std::conditional;
using std::is_same;
using std::pair;
using std::tie;
using std::vector;

namespace arrayfire {
namespace cuda {

#ifdef WITH_CUDNN

auto getLogger() { return getCudnnPlugin().getLogger(); }

template<typename Desc, typename T>
auto toCudnn(Array<T> arr) {
    auto descriptor = make_handle<Desc>();
    cudnnSet(descriptor, getCudnnDataType<T>(), arr.dims());
    return descriptor;
}

template<typename T>
using scale_type =
    typename conditional<is_same<T, double>::value, double, float>::type;

pair<cudnnConvolutionFwdAlgo_t, size_t> getForwardAlgorithm(
    cudnnHandle_t cudnn, cudnnTensorDescriptor_t input_descriptor,
    cudnnFilterDescriptor_t filter_descriptor,
    cudnnConvolutionDescriptor_t convolution_descriptor,
    cudnnTensorDescriptor_t output_descriptor) {
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t workspace_bytes = 0;

    auto version = getCudnnPlugin().getVersion();
    if (version.major() >= 8) {
        int maxAlgoCount = 0;
        CUDNN_CHECK(cuda::cudnnGetConvolutionForwardAlgorithmMaxCount(
            cudnn, &maxAlgoCount));

        vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(maxAlgoCount);
        int returnAlgoCount = 0;
        CUDNN_CHECK(cuda::cudnnFindConvolutionForwardAlgorithm(
            cudnn, input_descriptor, filter_descriptor, convolution_descriptor,
            output_descriptor, maxAlgoCount, &returnAlgoCount,
            perfResults.data()));

        for (int i = 0; i < returnAlgoCount; ++i) {
            if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
                convolution_algorithm = perfResults[i].algo;
                workspace_bytes       = perfResults[i].memory;
                break;
            }
        }
    } else {
        const int memory_limit =
            0;  // TODO: set to remaining space in memory manager?
        CUDNN_CHECK(cuda::cudnnGetConvolutionForwardAlgorithm(
            cudnn, input_descriptor, filter_descriptor, convolution_descriptor,
            output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            memory_limit, &convolution_algorithm));
        CUDNN_CHECK(cuda::cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_descriptor, filter_descriptor, convolution_descriptor,
            output_descriptor, convolution_algorithm, &workspace_bytes));
    }

    return {convolution_algorithm, workspace_bytes};
}

template<typename T>
Array<T> convolve2_cudnn(const Array<T> &signal, const Array<T> &filter,
                         const dim4 &stride, const dim4 &padding,
                         const dim4 &dilation) {
    cudnnHandle_t cudnn = nnHandle();

    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();
    auto input_descriptor       = toCudnn<cudnnTensorDescriptor_t>(signal);
    auto filter_descriptor      = toCudnn<cudnnFilterDescriptor_t>(filter);

    // create convolution descriptor
    auto convolution_descriptor = make_handle<cudnnConvolutionDescriptor_t>();

    CUDNN_CHECK(cuda::cudnnSetConvolution2dDescriptor(
        convolution_descriptor, padding[1], padding[0], stride[1], stride[0],
        dilation[1], dilation[0], CUDNN_CONVOLUTION, cudnn_dtype));

    // get output dimensions
    const int tensorDims = 4;
    int convolved_output_dim[tensorDims];
    CUDNN_CHECK(cuda::cudnnGetConvolutionNdForwardOutputDim(
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
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t workspace_bytes = 0;

    tie(convolution_algorithm, workspace_bytes) =
        getForwardAlgorithm(cudnn, input_descriptor, filter_descriptor,
                            convolution_descriptor, output_descriptor);

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    auto alpha = scalar<scale_type<T>>(1.0);
    auto beta  = scalar<scale_type<T>>(0.0);
    CUDNN_CHECK(cuda::cudnnConvolutionForward(
        cudnn, &alpha, input_descriptor, signal.device(), filter_descriptor,
        filter.device(), convolution_descriptor, convolution_algorithm,
        (void *)workspace_buffer.get(), workspace_bytes, &beta,
        output_descriptor, out.device()));

    return out;
}

template<typename T>
constexpr void checkTypeSupport() {
    static_assert(std::is_same<float, T>::value ||
                      std::is_same<double, T>::value ||
                      std::is_same<half, T>::value,
                  "Invalid CuDNN data type: only f64, f32, f16 are supported");
}

#endif

template<typename T>
Array<T> convolve2_base(const Array<T> &signal, const Array<T> &filter,
                        const dim4 &stride, const dim4 &padding,
                        const dim4 &dilation) {
    dim4 sDims = signal.dims();
    dim4 fDims = filter.dims();

    dim_t outputWidth =
        1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) /
                stride[0];
    dim_t outputHeight =
        1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) /
                stride[1];

    const bool retCols = false;
    Array<T> unwrapped =
        unwrap(signal, fDims[0], fDims[1], stride[0], stride[1], padding[0],
               padding[1], dilation[0], dilation[1], retCols);

    unwrapped  = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped =
        modDims(unwrapped, dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsedFilter = filter;

    collapsedFilter = flip(collapsedFilter, {1, 1, 0, 0});
    collapsedFilter = modDims(collapsedFilter,
                              dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    T alpha        = scalar<T>(1.0);
    T beta         = scalar<T>(0.0);
    const int Mdim = 1;
    const int Ndim = 1;
    Array<T> res   = createEmptyArray<T>(
        dim4(unwrapped.dims()[Mdim], collapsedFilter.dims()[Ndim],
               unwrapped.dims()[2], unwrapped.dims()[3]));
    gemm(res, AF_MAT_TRANS, AF_MAT_NONE, &alpha, unwrapped, collapsedFilter,
         &beta);
    res = modDims(res, dim4(outputWidth, outputHeight, signal.dims()[3],
                            collapsedFilter.dims()[1]));
    Array<T> out = reorder(res, dim4(0, 1, 3, 2));

    return out;
}

template<typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
#ifdef WITH_CUDNN
    if (getCudnnPlugin().isLoaded()) {
        checkTypeSupport<T>();
        return convolve2_cudnn<T>(signal, filter, stride, padding, dilation);
    }
#endif
    return convolve2_base<T>(signal, filter, stride, padding, dilation);
}

#define INSTANTIATE(T)                                                        \
    template Array<T> convolve2<T>(Array<T> const &signal,                    \
                                   Array<T> const &filter, const dim4 stride, \
                                   const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(half)
#undef INSTANTIATE

template<typename T>
Array<T> data_gradient_base(const Array<T> &incoming_gradient,
                            const Array<T> &original_signal,
                            const Array<T> &original_filter,
                            const Array<T> &convolved_output, af::dim4 stride,
                            af::dim4 padding, af::dim4 dilation) {
    UNUSED(convolved_output);
    const dim4 &cDims = incoming_gradient.dims();
    const dim4 &sDims = original_signal.dims();
    const dim4 &fDims = original_filter.dims();

    Array<T> collapsed_filter = original_filter;

    collapsed_filter = flip(collapsed_filter, {1, 1, 0, 0});
    collapsed_filter = modDims(collapsed_filter,
                               dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<T> collapsed_gradient = incoming_gradient;
    collapsed_gradient          = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient          = modDims(
        collapsed_gradient, dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    T alpha        = scalar<T>(1.0);
    T beta         = scalar<T>(0.0);
    const int Mdim = 0;
    const int Ndim = 0;
    Array<T> res   = createEmptyArray<T>(
        dim4(collapsed_gradient.dims()[Mdim], collapsed_filter.dims()[Ndim],
               collapsed_gradient.dims()[3], collapsed_gradient.dims()[3]));
    gemm(res, AF_MAT_NONE, AF_MAT_TRANS, &alpha, collapsed_gradient,
         collapsed_filter, &beta);
    res = modDims(res, dim4(res.dims()[0] / sDims[3], sDims[3],
                            fDims[0] * fDims[1], sDims[2]));
    res = reorder(res, dim4(0, 2, 3, 1));

    const bool retCols = false;
    res = wrap_dilated(res, sDims[0], sDims[1], fDims[0], fDims[1], stride[0],
                       stride[1], padding[0], padding[1], dilation[0],
                       dilation[1], retCols);

    return res;
}

#ifdef WITH_CUDNN
template<typename T>
Array<T> data_gradient_cudnn(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<T> &original_filter,
                             const Array<T> &convolved_output, af::dim4 stride,
                             af::dim4 padding, af::dim4 dilation) {
    UNUSED(convolved_output);
    auto cudnn = nnHandle();

    dim4 sDims = original_signal.dims();
    dim4 fDims = original_filter.dims();

    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();

    // create x descriptor
    auto dx_descriptor = toCudnn<cudnnTensorDescriptor_t>(original_signal);
    auto dy_descriptor = toCudnn<cudnnTensorDescriptor_t>(incoming_gradient);

    // create output filter gradient descriptor
    auto w_descriptor = make_handle<cudnnFilterDescriptor_t>();

    CUDNN_CHECK(cuda::cudnnSetFilter4dDescriptor(w_descriptor, cudnn_dtype,
                                                 CUDNN_TENSOR_NCHW, fDims[3],
                                                 fDims[2], fDims[1], fDims[0]));

    // create convolution descriptor
    auto convolution_descriptor = make_handle<cudnnConvolutionDescriptor_t>();

    CUDNN_CHECK(cuda::cudnnSetConvolution2dDescriptor(
        convolution_descriptor, padding[1], padding[0], stride[1], stride[0],
        dilation[1], dilation[0], CUDNN_CONVOLUTION, cudnn_dtype));

    cudnnConvolutionBwdDataAlgo_t bwd_data_convolution_algorithm;
    if ((dilation[0] == 1 && dilation[1] == 1) || is_same<T, half>::value) {
        bwd_data_convolution_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else {
        bwd_data_convolution_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    }

    // figure out scratch space memory requirements
    size_t workspace_bytes;
    CUDNN_CHECK(cuda::cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn, w_descriptor, dy_descriptor, convolution_descriptor,
        dx_descriptor, bwd_data_convolution_algorithm, &workspace_bytes));

    dim4 odims(sDims[0], sDims[1], sDims[2], sDims[3]);
    Array<T> out = createEmptyArray<T>(odims);

    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    auto alpha = scalar<scale_type<T>>(1.0);
    auto beta  = scalar<scale_type<T>>(0.0);

    CUDNN_CHECK(cuda::cudnnConvolutionBackwardData(
        cudnn, &alpha, w_descriptor, original_filter.get(), dy_descriptor,
        incoming_gradient.get(), convolution_descriptor,
        bwd_data_convolution_algorithm, (void *)workspace_buffer.get(),
        workspace_bytes, &beta, dx_descriptor, out.device()));

    return out;
}
#endif

template<typename T>
Array<T> conv2DataGradient(const Array<T> &incoming_gradient,
                           const Array<T> &original_signal,
                           const Array<T> &original_filter,
                           const Array<T> &convolved_output, af::dim4 stride,
                           af::dim4 padding, af::dim4 dilation) {
#ifdef WITH_CUDNN
    if (getCudnnPlugin().isLoaded()) {
        checkTypeSupport<T>();
        return data_gradient_cudnn<T>(incoming_gradient, original_signal,
                                      original_filter, convolved_output, stride,
                                      padding, dilation);
    }
#endif
    return data_gradient_base<T>(incoming_gradient, original_signal,
                                 original_filter, convolved_output, stride,
                                 padding, dilation);
}

template<typename T>
Array<T> filter_gradient_base(const Array<T> &incoming_gradient,
                              const Array<T> &original_signal,
                              const Array<T> &original_filter,
                              const Array<T> &convolved_output, af::dim4 stride,
                              af::dim4 padding, af::dim4 dilation) {
    UNUSED(convolved_output);
    const dim4 &cDims = incoming_gradient.dims();
    const dim4 &fDims = original_filter.dims();

    const bool retCols = false;
    Array<T> unwrapped =
        unwrap(original_signal, fDims[0], fDims[1], stride[0], stride[1],
               padding[0], padding[1], dilation[0], dilation[1], retCols);

    unwrapped  = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped =
        modDims(unwrapped, dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsed_gradient = incoming_gradient;
    collapsed_gradient          = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient          = modDims(
        collapsed_gradient, dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    T alpha        = scalar<T>(1.0);
    T beta         = scalar<T>(0.0);
    const int Mdim = 0;
    const int Ndim = 1;
    Array<T> res   = createEmptyArray<T>(
        dim4(unwrapped.dims()[Mdim], collapsed_gradient.dims()[Ndim],
               unwrapped.dims()[2], unwrapped.dims()[3]));
    gemm(res, AF_MAT_NONE, AF_MAT_NONE, &alpha, unwrapped, collapsed_gradient,
         &beta);
    res = modDims(res, dim4(fDims[0], fDims[1], fDims[2], fDims[3]));

    return flip(res, {1, 1, 0, 0});
}

#ifdef WITH_CUDNN

pair<cudnnConvolutionBwdFilterAlgo_t, size_t> getBackwardFilterAlgorithm(
    cudnnHandle_t cudnn, cudnnTensorDescriptor_t x_descriptor,
    cudnnTensorDescriptor_t dy_descriptor,
    cudnnConvolutionDescriptor_t convolution_descriptor,
    cudnnFilterDescriptor_t dw_descriptor) {
    // determine algorithm to use
    cudnnConvolutionBwdFilterAlgo_t bwd_filt_convolution_algorithm;
    // figure out scratch space memory requirements
    size_t workspace_bytes = 0;

    auto version = getCudnnPlugin().getVersion();
    if (version.major() >= 8) {
        int maxAlgoCount = 0;
        CUDNN_CHECK(cuda::cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
            cudnn, &maxAlgoCount));

        vector<cudnnConvolutionBwdFilterAlgoPerf_t> perfResults(maxAlgoCount);
        int returnAlgoCount = 0;
        CUDNN_CHECK(cuda::cudnnFindConvolutionBackwardFilterAlgorithm(
            cudnn, x_descriptor, dy_descriptor, convolution_descriptor,
            dw_descriptor, maxAlgoCount, &returnAlgoCount, perfResults.data()));

        for (int i = 0; i < returnAlgoCount; ++i) {
            if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
                bwd_filt_convolution_algorithm = perfResults[i].algo;
                workspace_bytes                = perfResults[i].memory;
                break;
            }
        }
    } else {
        CUDNN_CHECK(cuda::cudnnGetConvolutionBackwardFilterAlgorithm(
            cudnn, x_descriptor, dy_descriptor, convolution_descriptor,
            dw_descriptor, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
            &bwd_filt_convolution_algorithm));
        CUDNN_CHECK(cuda::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cudnn, x_descriptor, dy_descriptor, convolution_descriptor,
            dw_descriptor, bwd_filt_convolution_algorithm, &workspace_bytes));
    }
    return {bwd_filt_convolution_algorithm, workspace_bytes};
}

template<typename T>
Array<T> filter_gradient_cudnn(const Array<T> &incoming_gradient,
                               const Array<T> &original_signal,
                               const Array<T> &original_filter,
                               const Array<T> &convolved_output,
                               af::dim4 stride, af::dim4 padding,
                               af::dim4 dilation) {
    UNUSED(convolved_output);
    auto cudnn = nnHandle();

    const dim4 &fDims = original_filter.dims();

    // create dx descriptor
    cudnnDataType_t cudnn_dtype = getCudnnDataType<T>();
    auto x_descriptor  = toCudnn<cudnnTensorDescriptor_t>(original_signal);
    auto dy_descriptor = toCudnn<cudnnTensorDescriptor_t>(incoming_gradient);

    // create convolution descriptor
    auto convolution_descriptor = make_handle<cudnnConvolutionDescriptor_t>();

    CUDNN_CHECK(cuda::cudnnSetConvolution2dDescriptor(
        convolution_descriptor, padding[1], padding[0], stride[1], stride[0],
        dilation[1], dilation[0], CUDNN_CONVOLUTION, cudnn_dtype));

    // create output filter gradient descriptor
    auto dw_descriptor = toCudnn<cudnnFilterDescriptor_t>(original_filter);

    // determine algorithm to use
    cudnnConvolutionBwdFilterAlgo_t bwd_filt_convolution_algorithm;
    // figure out scratch space memory requirements
    size_t workspace_bytes = 0;

    tie(bwd_filt_convolution_algorithm, workspace_bytes) =
        getBackwardFilterAlgorithm(cudnn, x_descriptor, dy_descriptor,
                                   convolution_descriptor, dw_descriptor);

    // prepare output array and scratch space
    Array<T> out          = createEmptyArray<T>(fDims);
    auto workspace_buffer = memAlloc<char>(workspace_bytes);

    // perform convolution
    auto alpha = scalar<scale_type<T>>(1.0);
    auto beta  = scalar<scale_type<T>>(0.0);
    CUDNN_CHECK(cuda::cudnnConvolutionBackwardFilter(
        cudnn, &alpha, x_descriptor, original_signal.device(), dy_descriptor,
        incoming_gradient.device(), convolution_descriptor,
        bwd_filt_convolution_algorithm, (void *)workspace_buffer.get(),
        workspace_bytes, &beta, dw_descriptor, out.device()));

    return out;
}
#endif

template<typename T>
Array<T> conv2FilterGradient(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<T> &original_filter,
                             const Array<T> &convolved_output, af::dim4 stride,
                             af::dim4 padding, af::dim4 dilation) {
#ifdef WITH_CUDNN
    if (getCudnnPlugin().isLoaded()) {
        checkTypeSupport<T>();
        return filter_gradient_cudnn<T>(incoming_gradient, original_signal,
                                        original_filter, convolved_output,
                                        stride, padding, dilation);
    }
#endif
    return filter_gradient_base<T>(incoming_gradient, original_signal,
                                   original_filter, convolved_output, stride,
                                   padding, dilation);
}

#define INSTANTIATE(T)                                                      \
    template Array<T> conv2DataGradient<T>(                                 \
        Array<T> const &incoming_gradient, Array<T> const &original_signal, \
        Array<T> const &original_filter, Array<T> const &convolved_output,  \
        const dim4 stride, const dim4 padding, const dim4 dilation);        \
    template Array<T> conv2FilterGradient<T>(                               \
        Array<T> const &incoming_gradient, Array<T> const &original_signal, \
        Array<T> const &original_filter, Array<T> const &convolved_output,  \
        const dim4 stride, const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(half)
#undef INSTANTIATE

}  // namespace cuda
}  // namespace arrayfire
