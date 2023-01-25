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
#include <backend.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <dims_param.hpp>
#include <nvrtc_kernel_headers/copy_cuh.hpp>
#include <nvrtc_kernel_headers/memcopy_cuh.hpp>
#include <threadsMgt.hpp>

#include <algorithm>

namespace arrayfire {
namespace cuda {
namespace kernel {

// Increase vectorization by increasing the used type up to maxVectorWidth.
// Example:
//  input array<int> with return value = 4, means that the array became
//  array<int4>.
//
// Parameters
//  - IN     maxVectorWidth: maximum vectorisation desired
//  - IN/OUT dims[4]: dimensions of the array
//  - IN/OUT istrides[4]: strides of the input array
//  - IN/OUT indims: ndims of the input array.  Updates when dim[0] becomes 1
//  - IN/OUT ioffset: offset of the input array
//  - IN/OUT ostrides[4]: strides of the output array
//  - IN/OUT ooffset: offset of the output array
//
// Returns
//  - maximum obtained vectorization.
//  - All the parameters are updated accordingly
//
template<typename T>
dim_t vectorizeShape(const dim_t maxVectorWidth, Param<T> &out, dim_t &indims,
                     CParam<T> &in) {
    dim_t vectorWidth{1};
    if ((maxVectorWidth != 1) & (in.strides[0] == 1) & (out.strides[0] == 1)) {
        // Only adjacent items can be grouped into a base vector type
        void *in_ptr{(void *)in.ptr};
        void *out_ptr{(void *)out.ptr};
        // - global is the OR of the values to be checked.  When global is
        // divisable by 2, than all source values are also
        dim_t global{in.dims[0]};
        for (int i{1}; i < indims; ++i) {
            global |= in.strides[i] | out.strides[i];
        }
        // - The buffers are always aligned at 128 Bytes.  The pointers in the
        // Param<T> structure are however, direct pointers (including the
        // offset), so the final pointer has to be chedked on alignment
        size_t filler{64};  // give enough space for the align to move
        unsigned count{0};
        while (((global & 1) == 0) & (vectorWidth < maxVectorWidth) &&
               (in.ptr ==
                std::align(alignof(T) * vectorWidth * 2, 1, in_ptr, filler)) &&
               (out.ptr ==
                std::align(alignof(T) * vectorWidth * 2, 1, out_ptr, filler))) {
            ++count;
            vectorWidth <<= 1;
            global >>= 1;
        }
        if (count != 0) {
            // update the dimensions, to compensate for the vector base
            // type change
            in.dims[0] >>= count;
            for (int i{1}; i < indims; ++i) {
                in.strides[i] >>= count;
                out.strides[i] >>= count;
            }
            if (in.dims[0] == 1) {
                // Vectorization has absorbed the full dim0, so eliminate
                // this dimension
                --indims;
                for (int i{0}; i < indims; ++i) {
                    in.dims[i]     = in.dims[i + 1];
                    in.strides[i]  = in.strides[i + 1];
                    out.strides[i] = out.strides[i + 1];
                }
                in.dims[indims] = 1;
            }
        }
    }
    return vectorWidth;
}

template<typename T>
void memcopy(Param<T> out, CParam<T> in, dim_t indims) {
    const size_t totalSize{in.elements() * sizeof(T) * 2};
    removeEmptyColumns(in.dims, indims, out.strides);
    indims = removeEmptyColumns(in.dims, indims, in.dims, in.strides);
    indims = combineColumns(in.dims, in.strides, indims, out.strides);

    // Optimization memory access and caching.
    // Best performance is achieved with the highest vectorization
    // (<int> --> <int2>,<int4>, ...), since more data is processed per IO.

    // 16 Bytes gives best performance (=cdouble)
    const dim_t maxVectorWidth{sizeof(T) > 8 ? 1 : 16 / sizeof(T)};
    const dim_t vectorWidth{vectorizeShape(maxVectorWidth, out, indims, in)};
    const size_t sizeofNewT{sizeof(T) * vectorWidth};

    threadsMgt<dim_t> th(in.dims, indims);
    const dim3 threads{th.genThreads()};
    const dim3 blocks{th.genBlocks(threads, 1, 1, totalSize, sizeofNewT)};

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    // select the kernel with the necessary loopings
    const char *kernelName{th.loop0   ? "arrayfire::cuda::memCopyLoop0"
                           : th.loop2 ? "arrayfire::cuda::memCopyLoop123"
                           : th.loop1 ? th.loop3
                                            ? "arrayfire::cuda::memCopyLoop13"
                                            : "arrayfire::cuda::memCopyLoop1"
                           : th.loop3 ? "arrayfire::cuda::memCopyLoop3"
                                      : "arrayfire::cuda::memCopy"};

    // Conversion to cuda base vector types.
    switch (sizeofNewT) {
        case 1: {
            auto memCopy{common::getKernel(kernelName, {{memcopy_cuh_src}},
                                           TemplateArgs(TemplateArg("char")))};
            memCopy(qArgs, Param<char>((char *)out.ptr, out.dims, out.strides),
                    CParam<char>((const char *)in.ptr, in.dims, in.strides));
        } break;
        case 2: {
            auto memCopy{common::getKernel(kernelName, {{memcopy_cuh_src}},
                                           TemplateArgs(TemplateArg("short")))};
            memCopy(qArgs,
                    Param<short>((short *)out.ptr, out.dims, out.strides),
                    CParam<short>((const short *)in.ptr, in.dims, in.strides));
        } break;
        case 4: {
            auto memCopy{common::getKernel(kernelName, {{memcopy_cuh_src}},
                                           TemplateArgs(TemplateArg("float")))};
            memCopy(qArgs,
                    Param<float>((float *)out.ptr, out.dims, out.strides),
                    CParam<float>((const float *)in.ptr, in.dims, in.strides));
        } break;
        case 8: {
            auto memCopy{
                common::getKernel(kernelName, {{memcopy_cuh_src}},
                                  TemplateArgs(TemplateArg("float2")))};
            memCopy(
                qArgs, Param<float2>((float2 *)out.ptr, out.dims, out.strides),
                CParam<float2>((const float2 *)in.ptr, in.dims, in.strides));
        } break;
        case 16: {
            auto memCopy{
                common::getKernel(kernelName, {{memcopy_cuh_src}},
                                  TemplateArgs(TemplateArg("float4")))};
            memCopy(
                qArgs, Param<float4>((float4 *)out.ptr, out.dims, out.strides),
                CParam<float4>((const float4 *)in.ptr, in.dims, in.strides));
        } break;
        default: assert("type is larger than 16 bytes, which is unsupported");
    }
    POST_LAUNCH_CHECK();
}

template<typename inType, typename outType>
void copy(Param<outType> dst, CParam<inType> src, dim_t ondims,
          outType default_value, double factor) {
    const size_t totalSize{dst.elements() * sizeof(outType) +
                           src.elements() * sizeof(inType)};
    bool same_dims{true};
    for (dim_t i{0}; i < ondims; ++i) {
        if (src.dims[i] > dst.dims[i]) {
            src.dims[i] = dst.dims[i];
        } else if (src.dims[i] != dst.dims[i]) {
            same_dims = false;
        }
    }
    removeEmptyColumns(dst.dims, ondims, src.dims, src.strides);
    ondims = removeEmptyColumns(dst.dims, ondims, dst.dims, dst.strides);
    ondims =
        combineColumns(dst.dims, dst.strides, ondims, src.dims, src.strides);

    threadsMgt<dim_t> th(dst.dims, ondims);
    const dim3 threads{th.genThreads()};
    const dim3 blocks{th.genBlocks(threads, 1, 1, totalSize, sizeof(outType))};

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    auto copy{common::getKernel(
        th.loop0                 ? "arrayfire::cuda::scaledCopyLoop0"
        : (th.loop2 || th.loop3) ? "arrayfire::cuda::scaledCopyLoop123"
        : th.loop1               ? "arrayfire::cuda::scaledCopyLoop1"
                                 : "arrayfire::cuda::scaledCopy",
        {{copy_cuh_src}},
        TemplateArgs(TemplateTypename<inType>(), TemplateTypename<outType>(),
                     TemplateArg(same_dims), TemplateArg(factor != 1.0)))};

    copy(qArgs, dst, src, default_value, factor);

    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
