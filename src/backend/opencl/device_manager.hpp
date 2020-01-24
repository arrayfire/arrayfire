/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

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

namespace graphics {
class ForgeManager;
}

namespace common {
namespace memory {
class MemoryManagerBase;
}
} //arrayfire internal namespace

using common::memory::MemoryManagerBase;

namespace opencl {

// opencl namespace forward declarations
class GraphicsResourceManager;
struct kc_entry_t;  // kernel cache entry
class PlanCache;  // clfft

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

    friend graphics::ForgeManager& forgeManager();

    friend GraphicsResourceManager& interopManager();

    friend PlanCache& fftManager();

    friend void addKernelToCache(int device, const std::string& key,
                                 const kc_entry_t entry);

    friend void removeKernelFromCache(int device, const std::string& key);

    friend kc_entry_t kernelCache(int device, const std::string& key);

    friend std::string getDeviceInfo();

    friend int getDeviceCount();

    friend int getDeviceIdFromNativeId(cl_device_id id);

    friend const cl::Context& getContext();

    friend cl::CommandQueue& getQueue();

    friend const cl::Device& getDevice(int id);

    friend size_t getDeviceMemorySize(int device);

    friend bool isGLSharingSupported();

    friend bool isDoubleSupported(int device);

    friend bool isHalfSupported(int device);

    friend void devprop(char* d_name, char* d_platform, char* d_toolkit,
                        char* d_compute);

    friend int setDevice(int device);

    friend void addDeviceContext(cl_device_id dev, cl_context cxt,
                                 cl_command_queue que);

    friend void setDeviceContext(cl_device_id dev, cl_context cxt);

    friend void removeDeviceContext(cl_device_id dev, cl_context ctx);

    friend int getActiveDeviceType();

    friend int getActivePlatform();

   public:
    static const unsigned MAX_DEVICES = 32;

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
    std::vector<cl::Device*> mDevices;
    std::vector<cl::Context*> mContexts;
    std::vector<cl::CommandQueue*> mQueues;
    std::vector<bool> mIsGLSharingOn;
    std::vector<int> mDeviceTypes;
    std::vector<int> mPlatforms;
    unsigned mUserDeviceOffset;

    std::unique_ptr<graphics::ForgeManager> fgMngr;
    std::unique_ptr<MemoryManagerBase> memManager;
    std::unique_ptr<MemoryManagerBase> pinnedMemManager;
    std::unique_ptr<GraphicsResourceManager> gfxManagers[MAX_DEVICES];
    std::unique_ptr<clfftSetupData> mFFTSetup;
    std::mutex mutex;

    using BoostProgCache = boost::shared_ptr<boost::compute::program_cache>;
    std::vector<BoostProgCache*> mBoostProgCacheVector;
};

}  // namespace opencl
