/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(WITH_GRAPHICS)
#include <platform.hpp>
#include <common/InteropManager.hpp>
#include <map>
#include <vector>

namespace opencl
{
typedef cl::Buffer* CGR_t;

class GraphicsResourceManager : public common::InteropManager<GraphicsResourceManager, CGR_t>
{
    public:
        GraphicsResourceManager() {}

        std::vector<CGR_t> registerResources(std::vector<uint32_t> resources);
        void unregisterResource(CGR_t handle);

    protected:
        GraphicsResourceManager(GraphicsResourceManager const&);
        void operator=(GraphicsResourceManager const&);
};
}
#endif
