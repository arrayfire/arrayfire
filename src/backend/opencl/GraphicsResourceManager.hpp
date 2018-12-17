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
#include <common/InteropManager.hpp>

#include <map>
#include <vector>

namespace cl
{
class Buffer;
}

namespace opencl {
class GraphicsResourceManager :
    public common::InteropManager<GraphicsResourceManager, cl::Buffer>
{
    public:
        using ShrdResVector = std::vector< std::shared_ptr<cl::Buffer> >;

        GraphicsResourceManager() {}
        ShrdResVector registerResources(std::vector<uint32_t> resources);

    protected:
        GraphicsResourceManager(GraphicsResourceManager const&);
        void operator=(GraphicsResourceManager const&);
};
}
#endif
