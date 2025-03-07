/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cstddef>
#include <memory>

namespace spdlog {
class logger;
}
namespace arrayfire {
namespace common {

/**
 * An interface that provides backend-specific memory management functions,
 * typically calling a dedicated backend-specific native API. Stored, wrapped,
 * and called by a MemoryManagerBase, from which calls to its interface are
 * delegated.
 */
class AllocatorInterface {
   public:
    AllocatorInterface() = default;
    virtual ~AllocatorInterface() {}
    virtual void shutdown()                       = 0;
    virtual int getActiveDeviceId()               = 0;
    virtual size_t getMaxMemorySize(int id)       = 0;
    virtual void *nativeAlloc(const size_t bytes) = 0;
    virtual void nativeFree(void *ptr)            = 0;
    virtual spdlog::logger *getLogger() final { return this->logger.get(); }

   protected:
    std::shared_ptr<spdlog::logger> logger;
};

}  // namespace common
}  // namespace arrayfire
