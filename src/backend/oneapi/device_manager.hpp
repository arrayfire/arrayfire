/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <sycl/sycl.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifndef AF_ONEAPI_MEM_DEBUG
#define AF_ONEAPI_MEM_DEBUG 0
#endif

namespace spdlog {
class logger;
}

namespace arrayfire {
namespace common {
class ForgeManager;
class MemoryManagerBase;
}  // namespace common
}  // namespace arrayfire

using arrayfire::common::MemoryManagerBase;

namespace arrayfire {
namespace oneapi {

// opencl namespace forward declarations
class GraphicsResourceManager;
struct kc_entry_t;  // kernel cache entry

class DeviceManager {
    friend MemoryManagerBase& memoryManager();

    friend void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManager();

    void resetMemoryManager();

    friend MemoryManagerBase& pinnedMemoryManager();

    friend void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManagerPinned();

    void resetMemoryManagerPinned();

    friend arrayfire::common::ForgeManager& forgeManager();

    friend GraphicsResourceManager& interopManager();

    friend void addKernelToCache(int device, const std::string& key,
                                 const kc_entry_t entry);

    friend void removeKernelFromCache(int device, const std::string& key);

    friend kc_entry_t kernelCache(int device, const std::string& key);

    friend std::string getDeviceInfo() noexcept;

    friend int getDeviceCount() noexcept;

    // friend int getDeviceIdFromNativeId(cl_device_id id);

    friend const sycl::context& getContext();

    friend sycl::queue& getQueue();

    friend sycl::queue* getQueueHandle(int device_id);

    friend const sycl::device& getDevice(int id);

    friend const std::string& getActiveDeviceBaseBuildFlags();

    friend size_t getDeviceMemorySize(int device);

    friend bool isGLSharingSupported();

    friend bool isDoubleSupported(unsigned device);

    friend bool isHalfSupported(unsigned device);

    friend void devprop(char* d_name, char* d_platform, char* d_toolkit,
                        char* d_compute);

    friend int setDevice(int device);

    friend void addDeviceContext(sycl::device& dev, sycl::context& ctx,
                                 sycl::queue& que);

    friend void setDeviceContext(sycl::device& dev, sycl::context& ctx);

    friend void removeDeviceContext(sycl::device& dev, sycl::context& ctx);

    friend int getActiveDeviceType();

    friend int getActivePlatform();

   public:
    static const int MAX_DEVICES = 32;

    static DeviceManager& getInstance();

    ~DeviceManager();

    spdlog::logger* getLogger();

   protected:
    DeviceManager();

    // Following two declarations are required to
    // avoid copying accidental copy/assignment
    // of instance returned by getInstance to other
    // variables
    DeviceManager(DeviceManager const&);
    void operator=(DeviceManager const&);
    void markDeviceForInterop(const int device, const void* wHandle);

   private:
    // Attributes
    std::shared_ptr<spdlog::logger> logger;
    std::mutex deviceMutex;
    std::vector<std::unique_ptr<sycl::device>> mDevices;
    std::vector<std::unique_ptr<sycl::context>> mContexts;
    std::vector<std::unique_ptr<sycl::queue>> mQueues;
    std::vector<bool> mIsGLSharingOn;
    std::vector<std::string> mBaseOpenCLBuildFlags;
    std::vector<int> mDeviceTypes;
    std::vector<int> mPlatforms;
    unsigned mUserDeviceOffset;

    std::unique_ptr<arrayfire::common::ForgeManager> fgMngr;
    std::unique_ptr<MemoryManagerBase> memManager;
    std::unique_ptr<MemoryManagerBase> pinnedMemManager;
    std::unique_ptr<GraphicsResourceManager> gfxManagers[MAX_DEVICES];
    std::mutex mutex;

    // using BoostProgCache = boost::shared_ptr<boost::compute::program_cache>;
    // std::vector<BoostProgCache*> mBoostProgCacheVector;
};

}  // namespace oneapi
}  // namespace arrayfire
