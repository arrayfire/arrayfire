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
#include <string>
#include <vector>

#include <common/MemoryManager.hpp>
#include <platform.hpp>

using common::memory::MemoryManagerBase;

#ifndef AF_OPENCL_MEM_DEBUG
#define AF_OPENCL_MEM_DEBUG 0
#endif

// Forward declaration from clFFT.h
struct clfftSetupData_;

namespace opencl {

class DeviceManager {
    friend MemoryManagerBase& memoryManager();

    friend MemoryManagerBase& pinnedMemoryManager();

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
    common::mutex_t deviceMutex;
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

    using BoostProgCache = boost::shared_ptr<boost::compute::program_cache>;
    std::vector<BoostProgCache*> mBoostProgCacheVector;
};

}  // namespace opencl
