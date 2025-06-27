/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Kernel.hpp>

#include <platform.hpp>

namespace arrayfire {
namespace cuda {

Kernel::DevPtrType Kernel::getDevPtr(const char* name) {
    Kernel::DevPtrType out = 0;
    size_t size            = 0;
    CU_CHECK(cuModuleGetGlobal(&out, &size, this->getModuleHandle(), name));
    return out;
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    CU_CHECK(cuMemcpyDtoDAsync(dst, src, bytes, getActiveStream()));
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t srcXInBytes, size_t bytes) {
    CU_CHECK(cuMemcpyDtoDAsync(dst, src, bytes, getActiveStream()));
}

void Kernel::copyToReadOnly2D(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                              size_t srcXInBytes, size_t srcPitchInBytes,
                              size_t height, size_t widthInBytes) {
    CUDA_MEMCPY2D pCopy;
    pCopy.srcXInBytes   = srcXInBytes;
    pCopy.srcY          = 0;
    pCopy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopy.srcDevice     = src;
    pCopy.srcPitch      = srcPitchInBytes;

    pCopy.dstXInBytes   = 0;
    pCopy.dstY          = 0;
    pCopy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopy.dstDevice     = dst;
    pCopy.dstPitch      = widthInBytes;

    pCopy.WidthInBytes = widthInBytes;
    pCopy.Height       = height;
    // CUdeviceptr srcStart = srcDevice + srcY*srcPitch + srcXInBytes;
    // CUdeviceptr dstStart = dstDevice + dstY*dstPitch + dstXInBytes;

    CU_CHECK(cuMemcpy2DAsync(&pCopy, getActiveStream()));
}

void Kernel::copyToReadOnly3D(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                              size_t srcXInBytes, size_t srcPitchInBytes,
                              size_t srcHeight, size_t depth, size_t height,
                              size_t widthInBytes) {
    CUDA_MEMCPY3D pCopy;
    pCopy.srcXInBytes   = srcXInBytes;
    pCopy.srcY          = 0;
    pCopy.srcZ          = 0;
    pCopy.srcLOD        = 0;
    pCopy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopy.srcDevice     = src;
    pCopy.srcPitch      = srcPitchInBytes;
    pCopy.srcHeight     = srcHeight;

    pCopy.dstXInBytes   = 0;
    pCopy.dstY          = 0;
    pCopy.dstZ          = 0;
    pCopy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopy.dstDevice     = dst;
    pCopy.dstPitch      = widthInBytes;
    pCopy.dstHeight     = height;

    pCopy.WidthInBytes = widthInBytes;
    pCopy.Height       = height;
    pCopy.Depth        = depth;
    // CUdeviceptr srcStart =
    //      srcDevice + (srcZ*srcHeight+srcY)*srcPitch + srcXInBytes;
    // CUdeviceptr dstStart =
    //      dstDevice + (dstZ*dstHeight+dstY)*dstPitch + dstXInBytes;
    CU_CHECK(cuMemcpy3DAsync(&pCopy, getActiveStream()));
}

void Kernel::setFlag(Kernel::DevPtrType dst, int* scalarValPtr,
                     const bool syncCopy) {
    CU_CHECK(
        cuMemcpyHtoDAsync(dst, scalarValPtr, sizeof(int), getActiveStream()));
    if (syncCopy) { CU_CHECK(cuStreamSynchronize(getActiveStream())); }
}

int Kernel::getFlag(Kernel::DevPtrType src) {
    int retVal = 0;
    CU_CHECK(cuMemcpyDtoHAsync(&retVal, src, sizeof(int), getActiveStream()));
    CU_CHECK(cuStreamSynchronize(getActiveStream()));
    return retVal;
}

}  // namespace cuda
}  // namespace arrayfire
