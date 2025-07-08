/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Kernel.hpp>

#include <backend.hpp>
#include <cl2hpp.hpp>
#include <common/defines.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace opencl {

Kernel::DevPtrType Kernel::getDevPtr(const char* name) {
    UNUSED(name);
    return nullptr;
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    getQueue().enqueueCopyBuffer(*src, *dst, 0, 0, bytes);
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t srcXInBytes, size_t bytes) {
    getQueue().enqueueCopyBuffer(*src, *dst, srcXInBytes, 0, bytes);
}

void Kernel::copyToReadOnly2D(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                              size_t srcXInBytes, size_t srcPitchInBytes,
                              size_t height, size_t widthInBytes) {
    std::array<size_t, 3> src_origin = {srcXInBytes, 0, 0};
    size_t src_row_pitch             = {srcPitchInBytes};
    size_t src_slice_pitch           = {0};

    std::array<size_t, 3> dst_origin = {0, 0, 0};
    size_t dst_row_pitch             = {widthInBytes};
    size_t dst_slice_pitch           = {0};

    std::array<size_t, 3> region = {widthInBytes, height, 1};

    // offset in bytes =
    // src_origin[1]*src_row_pitch + src_origin[0]

    getQueue().enqueueCopyBufferRect(*src, *dst, src_origin, dst_origin, region,
                                     src_row_pitch, src_slice_pitch,
                                     dst_row_pitch, dst_slice_pitch);
}

void Kernel::copyToReadOnly3D(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                              size_t srcXInBytes, size_t srcPitchInBytes,
                              size_t srcHeight, size_t depth, size_t height,
                              size_t widthInBytes) {
    std::array<size_t, 3> src_origin = {srcXInBytes, 0, 0};
    size_t src_row_pitch             = {srcPitchInBytes};
    size_t src_slice_pitch           = {srcHeight * srcPitchInBytes};

    std::array<size_t, 3> dst_origin = {0, 0, 0};
    size_t dst_row_pitch             = {widthInBytes};
    size_t dst_slice_pitch           = {height * widthInBytes};

    std::array<size_t, 3> region = {widthInBytes, height, depth};

    // offset in bytes =
    // src_origin[2]*src_slice_pitch + src_origin[1]*src_row_pitch +
    // src_origin[0]

    getQueue().enqueueCopyBufferRect(*src, *dst, src_origin, dst_origin, region,
                                     src_row_pitch, src_slice_pitch,
                                     dst_row_pitch, dst_slice_pitch);
}

void Kernel::setFlag(Kernel::DevPtrType dst, int* scalarValPtr,
                     const bool syncCopy) {
    UNUSED(syncCopy);
    getQueue().enqueueFillBuffer(*dst, *scalarValPtr, 0, sizeof(int));
}

int Kernel::getFlag(Kernel::DevPtrType src) {
    int retVal = 0;
    getQueue().enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(int), &retVal);
    return retVal;
}

}  // namespace opencl
}  // namespace arrayfire
