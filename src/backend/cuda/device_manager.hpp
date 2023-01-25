/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <platform.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using arrayfire::common::MemoryManagerBase;

#ifndef AF_CUDA_MEM_DEBUG
#define AF_CUDA_MEM_DEBUG 0
#endif

namespace arrayfire {
namespace cuda {

struct cudaDevice_t {
    cudaDeviceProp prop;
    size_t flops;
    int nativeId;
};

int& tlocalActiveDeviceId();

bool checkDeviceWithRuntime(int runtime, std::pair<int, int> compute);

class DeviceManager {
   public:
    static const int MAX_DEVICES = 16;

    static bool checkGraphicsInteropCapability();

    static DeviceManager& getInstance();
    ~DeviceManager();

    spdlog::logger* getLogger();

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

    friend std::string getDeviceInfo(int device) noexcept;

    friend std::string getPlatformInfo() noexcept;

    friend std::string getDriverVersion() noexcept;

    friend std::string getCUDARuntimeVersion() noexcept;

    friend std::string getDeviceInfo() noexcept;

    friend int getDeviceCount();

    friend int getDeviceNativeId(int device);

    friend int getDeviceIdFromNativeId(int nativeId);

    friend cudaStream_t getStream(int device);

    friend int setDevice(int device);

    friend const cudaDeviceProp& getDeviceProp(int device);

    friend std::pair<int, int> getComputeCapability(const int device);

   private:
    DeviceManager();

    // Following two declarations are required to
    // avoid copying accidental copy/assignment
    // of instance returned by getInstance to other
    // variables
    DeviceManager(DeviceManager const&);
    void operator=(DeviceManager const&);

    // Attributes
    enum sort_mode { flops = 0, memory = 1, compute = 2, none = 3 };

    // Checks if the Graphics driver is capable of running the CUDA toolkit
    // version that ArrayFire was compiled against
    void checkCudaVsDriverVersion();
    void sortDevices(sort_mode mode = flops);

    int setActiveDevice(int device, int nId = -1);

    std::shared_ptr<spdlog::logger> logger;

    std::vector<cudaDevice_t> cuDevices;
    std::vector<std::pair<int, int>> devJitComputes;

    int nDevices;
    cudaStream_t streams[MAX_DEVICES]{};

    std::unique_ptr<arrayfire::common::ForgeManager> fgMngr;

    std::unique_ptr<MemoryManagerBase> memManager;

    std::unique_ptr<MemoryManagerBase> pinnedMemManager;

    std::unique_ptr<GraphicsResourceManager> gfxManagers[MAX_DEVICES];

    std::mutex mutex;
};

}  // namespace cuda
}  // namespace arrayfire
