/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/InteropManager.hpp>
#include <driver_types.h>

#include <map>
#include <vector>

namespace arrayfire {
namespace cuda {
class GraphicsResourceManager
    : public common::InteropManager<GraphicsResourceManager,
                                    cudaGraphicsResource_t> {
   public:
    using ShrdResVector = std::vector<std::shared_ptr<cudaGraphicsResource_t>>;

    GraphicsResourceManager() {}
    static ShrdResVector registerResources(
        const std::vector<uint32_t> &resources);

   protected:
    GraphicsResourceManager(GraphicsResourceManager const &);
    void operator=(GraphicsResourceManager const &);
};
}  // namespace cuda
}  // namespace arrayfire
