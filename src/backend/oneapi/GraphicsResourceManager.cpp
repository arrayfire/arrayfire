/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <GraphicsResourceManager.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace oneapi {
GraphicsResourceManager::ShrdResVector
GraphicsResourceManager::registerResources(
    const std::vector<uint32_t>& resources) {
    ShrdResVector output;
    return output;
}
}  // namespace oneapi
}  // namespace arrayfire
