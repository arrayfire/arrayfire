/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/Logger.hpp>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <string>

namespace arrayfire {
namespace cuda {
namespace kernel_logger {

inline auto getLogger() {
    static auto logger = common::loggerFactory("kernel");
    return logger;
}
}  // namespace kernel_logger
}  // namespace cuda
}  // namespace arrayfire

template<>
struct fmt::formatter<dim3> : fmt::formatter<std::string> {
    // parse is inherited from formatter<string_view>.
    template<typename FormatContext>
    auto format(dim3 c, FormatContext& ctx) {
        std::string name = fmt::format("{} {} {}", c.x, c.y, c.z);
        return formatter<std::string>::format(name, ctx);
    }
};

#define CUDA_LAUNCH_SMEM(fn, blks, thrds, smem_size, ...)                   \
    do {                                                                    \
        {                                                                   \
            using namespace arrayfire::cuda::kernel_logger;                 \
            AF_TRACE(                                                       \
                "Launching {}: Blocks: [{}] Threads: [{}] "                 \
                "Shared Memory: {}",                                        \
                #fn, blks, thrds, smem_size);                               \
        }                                                                   \
        fn<<<blks, thrds, smem_size, arrayfire::cuda::getActiveStream()>>>( \
            __VA_ARGS__);                                                   \
    } while (false)

#define CUDA_LAUNCH(fn, blks, thrds, ...) \
    CUDA_LAUNCH_SMEM(fn, blks, thrds, 0, __VA_ARGS__)

// FIXME: Add a special flag for debug
#ifndef NDEBUG

#define POST_LAUNCH_CHECK()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaStreamSynchronize(arrayfire::cuda::getActiveStream())); \
    } while (0)

#else

#define POST_LAUNCH_CHECK()                                                 \
    do {                                                                    \
        if (arrayfire::cuda::synchronize_calls()) {                         \
            CUDA_CHECK(                                                     \
                cudaStreamSynchronize(arrayfire::cuda::getActiveStream())); \
        } else {                                                            \
            CUDA_CHECK(cudaPeekAtLastError());                              \
        }                                                                   \
    } while (0)

#endif
