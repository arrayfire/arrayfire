/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)
#include <err_cuda.hpp>
#include <GraphicsResourceManager.hpp>

namespace cuda
{
ShrdResVector GraphicsResourceManager::registerResources(std::vector<uint32_t> resources)
{
    ShrdResVector output;

    auto deleter = [](CGR_t* handle) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(*handle));
        delete handle;
    };

    for (auto id: resources) {
        CGR_t r;
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&r, id, cudaGraphicsMapFlagsWriteDiscard));
        output.emplace_back(new CGR_t(r), deleter);
    }

    return output;
}
}
#endif
