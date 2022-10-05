/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/InteropManager.hpp>

#include <map>
#include <memory>
#include <vector>

namespace oneapi {
class GraphicsResourceManager
    : public common::InteropManager<GraphicsResourceManager, std::byte> {
   public:
    using ShrdResVector = std::vector<std::shared_ptr<std::byte>>;

    GraphicsResourceManager() {}
    static ShrdResVector registerResources(
        const std::vector<uint32_t>& resources);

   protected:
    GraphicsResourceManager(GraphicsResourceManager const&);
    void operator=(GraphicsResourceManager const&);
};
}  // namespace oneapi
