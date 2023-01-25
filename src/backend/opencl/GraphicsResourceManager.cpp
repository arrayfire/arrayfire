/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <GraphicsResourceManager.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace opencl {
GraphicsResourceManager::ShrdResVector
GraphicsResourceManager::registerResources(
    const std::vector<uint32_t>& resources) {
    ShrdResVector output;

    for (auto id : resources) {
        output.emplace_back(new cl::BufferGL(
            getContext(), CL_MEM_WRITE_ONLY,  // NOLINT(hicpp-signed-bitwise)
            id, NULL));
    }

    return output;
}
}  // namespace opencl
}  // namespace arrayfire
