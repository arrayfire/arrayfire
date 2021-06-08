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
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <dims_param.hpp>
#include <nvrtc_kernel_headers/copy_cuh.hpp>
#include <nvrtc_kernel_headers/memcopy_cuh.hpp>
#include <algorithm>
#include <array>
#include <tuple>

namespace cuda {
namespace kernel {

// Push all linear columns to the front, to improve occupancy
// In case of copy operation, input & output parameters needs to be provided!!
// main dims:       mdims, mstrides, mndims
// optional dims:   odims, ostrides
// ALL parameters will be updated!!
template<bool RESHAPE = false>
void serializeArray(dim_t mdims[4], dim_t mstrides[4], dim_t &mndims,
                    dim_t ostrides[4], dim_t odims[4] = nullptr) noexcept {
    if (RESHAPE) assert(odims != nullptr);
    for (int c = 0; c < mndims - 1; c++) {
        if (mdims[c] == 1) {
            // Columns with 1 can always be removed.
            // strides of the last column are not updated, because:
            //    - dimension always becomes 1
            //    - strides are therefor not used
            for (int i = c; i < mndims - 1; ++i) {
                mdims[i] = mdims[i + 1];
                if (RESHAPE) odims[i] = odims[i + 1];
                mstrides[i] = mstrides[i + 1];
                ostrides[i] = ostrides[i + 1];
            }
            --mndims;
            mdims[mndims] = 1;
            if (RESHAPE) odims[mndims] = 1;
            --c;  // Redo this column, since it is removed now
        } else if (mdims[c] * mstrides[c] == mstrides[c + 1] &&
                   mdims[c] * ostrides[c] == ostrides[c + 1]) {
            // Combine columns, since they are linear
            // This will increase the dimension of the resulting column,
            // given more opportunities for kernel optimization
            mdims[c] *= mdims[c + 1];
            if (RESHAPE) odims[c] *= odims[c + 1];
            for (int i = c + 1; i < mndims - 1; ++i) {
                mdims[i] = mdims[i + 1];
                if (RESHAPE) odims[i] = odims[i + 1];
                mstrides[i] = mstrides[i + 1];
                ostrides[i] = ostrides[i + 1];
            }
            --mndims;
            mdims[mndims] = 1;
            if (RESHAPE) odims[mndims] = 1;
            --c;  // Redo this colum, since it is removed now
        }
    }
}

// To increase the workload inside a kernel, we move a part of the ndims
// dimension to the last one.  Since this not covered by WG or warp, this is
// always executed as an internal loop.
// In case of copy operation, input & output parameters needs to be provided!!
// main dims: mdims, mstrides, mndims
// optional dims: odims, ostrides
// ALL parameters will be updated!!
void inline increaseWorkload(dim_t elements, dim_t mdims[4], dim_t mstrides[4],
                             dim_t &mndims, dim_t ostrides[4]) noexcept {
    if (elements >= 8192 * 2 && mndims != AF_MAX_DIMS && mndims != 0) {
        // Start only increasing the workload, when all available threads are
        // occupied.

        // list is sorted according to performance improvement
        // 3x looping is faster than 4x, 2x remains faster than no looping
        for (const dim_t i : {3, 4, 5, 7, 11, 2}) {
            if (elements >= 8192 * i && (mdims[mndims - 1] % i) == 0) {
                mdims[mndims - 1] /= i;
                mdims[AF_MAX_DIMS - 1] = i;
                const dim_t mstride = mdims[mndims - 1] * mstrides[mndims - 1];
                const dim_t ostride = mdims[mndims - 1] * ostrides[mndims - 1];
                for (dim_t c = mndims; c < AF_MAX_DIMS; ++c) {
                    mstrides[c] = mstride;
                    ostrides[c] = ostride;
                }
                mndims = AF_MAX_DIMS;
                break;  // Perform this operation only once.
            }
        }
    }
}

template<typename T>
void memcopy(Param<T> out, CParam<T> in, dim_t indims) {
    bool isLinear  = true;
    dim_t elements = (indims == 0) ? 0 : 1;
    for (dim_t dim = 0; dim < indims; ++dim) {
        isLinear &=
            (elements == in.strides[dim]) & (elements == out.strides[dim]);
        elements *= in.dims[dim];
    }
    if (elements > 0) {
        const auto stream = cuda::getActiveStream();
        if (isLinear) {
            CUDA_CHECK(cudaMemcpyAsync(out.ptr, in.ptr,
                                       in.elements() * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            const int *maxGridSize =
                cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize;
            serializeArray(in.dims, in.strides, indims, out.strides);
            increaseWorkload(elements, in.dims, in.strides, indims,
                             out.strides);
            const dim3 threads = bestBlockSize<dim3>(in.dims, 32);
            dim3 blocks(divup(static_cast<unsigned>(in.dims[0]), threads.x),
                        divup(static_cast<unsigned>(in.dims[1]), threads.y),
                        divup(static_cast<unsigned>(in.dims[2]), threads.z));
            const bool loop1 = blocks.y > static_cast<unsigned>(maxGridSize[1]);
            if (loop1) blocks.y = maxGridSize[1];
            const bool loop2 = blocks.z > static_cast<unsigned>(maxGridSize[2]);
            if (loop2) blocks.z = maxGridSize[2];
            EnqueueArgs qArgs(blocks, threads, stream);

            auto memCopy = common::getKernel("cuda::memcopy", {memcopy_cuh_src},
                                             {
                                                 TemplateTypename<T>(),
                                                 TemplateArg(loop1),
                                                 TemplateArg(loop2),
                                             });
            memCopy(qArgs, out, in);
            POST_LAUNCH_CHECK();
        }
    }
}

template<typename T>
class CParamPlus {
   public:
    CParam<T> cparam;
    dim_t ooffset;
    CParamPlus(CParam<T> arr, dim_t off) : cparam(arr), ooffset(off){};
};

template<typename T>
void memcopyN(Param<T> out, std::vector<CParamPlus<T>> &Ins) {
    Kernel memCopy;
    bool loadKernel = true;
    bool loadLoop1 = false, loadLoop2 = false;
    const auto stream = cuda::getActiveStream();
    const int *maxGridSize =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize;

    for (auto &in_ : Ins) {
        dim_t indims_ = in_.cparam.dims[3] > 1   ? 4
                        : in_.cparam.dims[2] > 1 ? 3
                        : in_.cparam.dims[1] > 1 ? 2
                        : in_.cparam.dims[0] > 0 ? 1
                                                 : 0;
        Param<T> out_(out);
        out_.ptr += in_.ooffset;
        bool isLinear  = true;
        dim_t elements = (indims_ == 0) ? 0 : 1;
        for (dim_t dim = 0; dim < indims_; ++dim) {
            isLinear &= (elements == in_.cparam.strides[dim]) &
                        (elements == out_.strides[dim]);
            elements *= in_.cparam.dims[dim];
        }
        if (elements > 0) {
            if (isLinear) {
                CUDA_CHECK(cudaMemcpyAsync(out_.ptr, in_.cparam.ptr,
                                           in_.cparam.elements() * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
            } else {
                serializeArray(in_.cparam.dims, in_.cparam.strides, indims_,
                               out_.strides);
                increaseWorkload(elements, in_.cparam.dims, in_.cparam.strides,
                                 indims_, out_.strides);
                const dim3 threads = bestBlockSize<dim3>(in_.cparam.dims, 32);
                dim3 blocks(
                    divup(static_cast<unsigned>(in_.cparam.dims[0]), threads.x),
                    divup(static_cast<unsigned>(in_.cparam.dims[1]), threads.y),
                    divup(static_cast<unsigned>(in_.cparam.dims[2]),
                          threads.z));
                const bool loop1 = blocks.y > (unsigned)maxGridSize[1];
                if (loop1) blocks.y = (unsigned)maxGridSize[1];
                const bool loop2 = blocks.z > (unsigned)maxGridSize[2];
                if (loop2) blocks.z = (unsigned)maxGridSize[2];
                const EnqueueArgs qArgs(blocks, threads, stream);

                if ((loop1 && !loadLoop1) || (loop2 && !loadLoop2) ||
                    loadKernel) {
                    loadLoop1 |= loop1;
                    loadLoop2 |= loop2;
                    memCopy =
                        common::getKernel("cuda::memcopy", {memcopy_cuh_src},
                                          {
                                              TemplateTypename<T>(),
                                              TemplateArg(loadLoop1),
                                              TemplateArg(loadLoop2),
                                          });
                    loadKernel = false;
                }

                memCopy(qArgs, out_, in_.cparam);
                POST_LAUNCH_CHECK();
            }
        }
    }
}

template<typename inType, typename outType>
void copy(Param<outType> dst, CParam<inType> src, dim_t ondims,
          outType default_value, double factor) {
    const int ondims_ = static_cast<int>(ondims);
    dim_t elements    = (ondims_ == 0) ? 0 : 1;
    for (int dim = 0; dim < ondims_; ++dim) { elements *= dst.dims[dim]; }
    if (elements > 0) {
        const int *maxGridSize =
            cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize;
        serializeArray<true>(dst.dims, dst.strides, ondims, src.strides,
                             src.dims);
        const dim3 threads = bestBlockSize<dim3>(dst.dims, 32);
        dim3 blocks(divup(static_cast<unsigned>(dst.dims[0]), threads.x),
                    divup(static_cast<unsigned>(dst.dims[1]), threads.y),
                    divup(static_cast<unsigned>(dst.dims[2]), threads.z));
        const bool loop1 = blocks.y > static_cast<unsigned>(maxGridSize[1]);
        if (loop1) blocks.y = maxGridSize[1];
        const bool loop2 = blocks.z > static_cast<unsigned>(maxGridSize[2]);
        if (loop2) blocks.z = maxGridSize[2];

        EnqueueArgs qArgs(blocks, threads, getActiveStream());

        const bool same_dims =
            ((src.dims[0] == dst.dims[0]) && (src.dims[1] == dst.dims[1]) &&
             (src.dims[2] == dst.dims[2]) && (src.dims[3] == dst.dims[3]));

        auto copy = common::getKernel("cuda::reshapeCopy", {copy_cuh_src},
                                      {
                                          TemplateTypename<inType>(),
                                          TemplateTypename<outType>(),
                                          TemplateArg(same_dims),
                                          TemplateArg(loop1),
                                          TemplateArg(loop2),
                                          TemplateArg(factor != 1.0),
                                      });
        copy(qArgs, dst, src, default_value, factor);

        POST_LAUNCH_CHECK();
    }
}
}  // namespace kernel
}  // namespace cuda
