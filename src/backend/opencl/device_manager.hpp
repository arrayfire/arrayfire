/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/opencl.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifndef AF_OPENCL_MEM_DEBUG
#define AF_OPENCL_MEM_DEBUG 0
#endif

// Forward declarations
struct clfftSetupData_;

namespace cl {
class CommandQueue;
class Context;
class Device;
}  // namespace cl

namespace boost {
template<typename T>
class shared_ptr;

namespace compute {
class program_cache;
}
}  // namespace boost

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
namespace opencl {

// opencl namespace forward declarations
class GraphicsResourceManager;
struct kc_entry_t;  // kernel cache entry
class PlanCache;    // clfft

class DeviceManager {
    friend arrayfire::common::MemoryManagerBase& memoryManager();

    friend void setMemoryManager(
        std::unique_ptr<arrayfire::common::MemoryManagerBase> mgr);

    void setMemoryManager(
        std::unique_ptr<arrayfire::common::MemoryManagerBase> mgr);

    friend void resetMemoryManager();

    void resetMemoryManager();

    friend arrayfire::common::MemoryManagerBase& pinnedMemoryManager();

    friend void setMemoryManagerPinned(
        std::unique_ptr<arrayfire::common::MemoryManagerBase> mgr);

    void setMemoryManagerPinned(
        std::unique_ptr<arrayfire::common::MemoryManagerBase> mgr);

    friend void resetMemoryManagerPinned();

    void resetMemoryManagerPinned();

    friend arrayfire::common::ForgeManager& forgeManager();

    friend GraphicsResourceManager& interopManager();

    friend PlanCache& fftManager();

    friend void addKernelToCache(int device, const std::string& key,
                                 const kc_entry_t entry);

    friend void removeKernelFromCache(int device, const std::string& key);

    friend kc_entry_t kernelCache(int device, const std::string& key);

    friend std::string getDeviceInfo() noexcept;

    friend int getDeviceCount() noexcept;

    friend int getDeviceIdFromNativeId(cl_device_id id);

    friend const cl::Context& getContext();

    friend cl::CommandQueue& getQueue(int device_id);

    friend cl_command_queue getQueueHandle(int device_id);

    friend const cl::Device& getDevice(int id);

    friend const std::string& getActiveDeviceBaseBuildFlags();

    friend size_t getDeviceMemorySize(int device);

    friend bool isGLSharingSupported();

    friend bool isDoubleSupported(unsigned device);

    friend bool isHalfSupported(unsigned device);

    friend void devprop(char* d_name, char* d_platform, char* d_toolkit,
                        char* d_compute);

    friend int setDevice(int device);

    friend void addDeviceContext(cl_device_id dev, cl_context ctx,
                                 cl_command_queue que);

    friend void setDeviceContext(cl_device_id dev, cl_context ctx);

    friend void removeDeviceContext(cl_device_id dev, cl_context ctx);

    friend int getActiveDeviceType();

    friend cl::Platform& getActivePlatform();

    friend afcl::platform getActivePlatformVendor();

    friend bool isDeviceBufferAccessible(int buf_device_id, int execution_id);

   public:
    static const int MAX_DEVICES = 32;

    static DeviceManager& getInstance();

    ~DeviceManager();

    spdlog::logger* getLogger();

   protected:
    using clfftSetupData = clfftSetupData_;

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
    std::vector<std::unique_ptr<cl::Device>> mDevices;
    std::vector<std::unique_ptr<cl::Context>> mContexts;
    std::vector<std::unique_ptr<cl::CommandQueue>> mQueues;
    std::vector<bool> mIsGLSharingOn;
    std::vector<std::string> mBaseBuildFlags;
    std::vector<int> mDeviceTypes;
    std::vector<std::pair<std::unique_ptr<cl::Platform>, afcl::platform>>
        mPlatforms;
    unsigned mUserDeviceOffset;

    std::unique_ptr<arrayfire::common::ForgeManager> fgMngr;
    std::unique_ptr<MemoryManagerBase> memManager;
    std::unique_ptr<MemoryManagerBase> pinnedMemManager;
    std::unique_ptr<GraphicsResourceManager> gfxManagers[MAX_DEVICES];
    std::unique_ptr<clfftSetupData> mFFTSetup;
    std::mutex mutex;

    using BoostProgCache = boost::shared_ptr<boost::compute::program_cache>;
    std::vector<BoostProgCache*> mBoostProgCacheVector;
};

}  // namespace opencl
}  // namespace arrayfire
