/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)
#include <platform.hpp>
#include <GraphicsResourceManager.hpp>

namespace opencl
{
std::vector<CGR_t> GraphicsResourceManager::registerResources(std::vector<uint32_t> resources)
{
    std::vector<CGR_t> output;

    for (auto id: resources) {
        CGR_t r = new cl::BufferGL(opencl::getContext(), CL_MEM_WRITE_ONLY, id, NULL);
        output.push_back(r);
    }

    return output;
}

void GraphicsResourceManager::unregisterResource(CGR_t handle)
{
    delete handle;
}
}
#endif
