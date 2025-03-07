/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace arrayfire {
namespace cuda {

///
/// EnqueueArgs is a kernel launch configuration composition object
///
/// This structure is an composition of various parameters that are
/// required to successfully launch a CUDA kernel.
///
struct EnqueueArgs {
    // TODO(pradeep): this can be easily templated
    // template<typename Queue, typename Event>
    dim3 mBlocks;                  ///< Number of blocks per grid/kernel-launch
    dim3 mThreads;                 ///< Number of threads per block
    CUstream mStream;              ///< CUDA stream to enqueue the kernel on
    unsigned int mSharedMemSize;   ///< Size(in bytes) of shared memory used
    std::vector<CUevent> mEvents;  ///< Events to wait for kernel execution

    ///
    /// \brief EnqueueArgs constructor
    ///
    /// \param[in] blks is number of blocks per grid
    /// \param[in] thrds is number of threads per block
    /// \param[in] stream is CUDA steam on which kernel has to be enqueued
    /// \param[in] sharedMemSize is number of bytes of shared memory allocation
    /// \param[in] events is list of events to wait for kernel execution
    ///
    EnqueueArgs(dim3 blks, dim3 thrds, CUstream stream = 0,
                const unsigned int sharedMemSize   = 0,
                const std::vector<CUevent> &events = {})
        : mBlocks(blks)
        , mThreads(thrds)
        , mStream(stream)
        , mSharedMemSize(sharedMemSize)
        , mEvents(events) {}
};

}  // namespace cuda
}  // namespace arrayfire
