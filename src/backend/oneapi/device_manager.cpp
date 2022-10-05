/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/graphics_common.hpp>

#include <GraphicsResourceManager.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <err_oneapi.hpp>
#include <platform.hpp>  //TODO: blas.hpp? y tho, also Array.hpp
//#include <errorcodes.hpp>
#include <memory.hpp>
#include <version.hpp>
#include <af/oneapi.h>
#include <af/version.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using std::begin;
using std::end;
using std::find;
using std::make_unique;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::vector;
using sycl::device;
using sycl::platform;

namespace oneapi {

static inline bool compare_default(const unique_ptr<sycl::device>& ldev,
                                   const unique_ptr<sycl::device>& rdev) {
    // TODO: update sorting criteria
    // select according to something applicable to oneapi backend
    auto l_mem = ldev->get_info<sycl::info::device::local_mem_size>();
    auto r_mem = rdev->get_info<sycl::info::device::local_mem_size>();
    return l_mem > r_mem;
}

DeviceManager::DeviceManager()
    : logger(common::loggerFactory("platform"))
    , mUserDeviceOffset(0)
    , fgMngr(nullptr) {
    vector<sycl::platform> platforms;
    try {
        platforms = sycl::platform::get_platforms();
    } catch (sycl::exception& err) {
        AF_ERROR(
            "No sycl platforms found on this system. Ensure you have "
            "installed the device driver as well as the runtime.",
            AF_ERR_RUNTIME);
    }

    fgMngr = std::make_unique<graphics::ForgeManager>();

    AF_TRACE("Found {} sycl platforms", platforms.size());
    // Iterate through platforms, get all available devices and store them
    for (auto& platform : platforms) {
        vector<sycl::device> current_devices;
        try {
            current_devices = platform.get_devices();
        } catch (sycl::exception& err) {
            printf("DeviceManager::DeviceManager() exception: %s\n",
                   err.what());
            throw;
        }
        AF_TRACE("Found {} devices on platform {}", current_devices.size(),
                 platform.get_info<sycl::info::platform::name>());

        for (auto& dev : current_devices) {
            mDevices.emplace_back(make_unique<sycl::device>(dev));
            AF_TRACE("Found device {} on platform {}",
                     dev.get_info<sycl::info::device::name>(),
                     platform.get_info<sycl::info::platform::name>());
        }
    }

    int nDevices = mDevices.size();
    AF_TRACE("Found {} sycl devices", nDevices);

    if (nDevices == 0) { AF_ERROR("No sycl devices found", AF_ERR_RUNTIME); }

    // Sort sycl devices based on default criteria
    stable_sort(mDevices.begin(), mDevices.end(), compare_default);

    auto devices = move(mDevices);
    mDevices.clear();

    // Create contexts and queues once the sort is done
    for (int i = 0; i < nDevices; i++) {
        try {
            mContexts.push_back(make_unique<sycl::context>(*devices[i]));
            mQueues.push_back(
                make_unique<sycl::queue>(*mContexts.back(), *devices[i]));
            mIsGLSharingOn.push_back(false);
            // TODO:
            // mDeviceTypes.push_back(getDeviceTypeEnum(*devices[i]));
            // mPlatforms.push_back(getPlatformEnum(*devices[i]));
            mDevices.emplace_back(std::move(devices[i]));
        } catch (sycl::exception& err) {
            AF_TRACE("Error creating context for device {} with error {}\n",
                     devices[i]->get_info<sycl::info::device::name>(),
                     err.what());
        }
    }
    nDevices = mDevices.size();

    bool default_device_set = false;
    string deviceENV        = getEnvVar("AF_ONEAPI_DEFAULT_DEVICE");
    if (!deviceENV.empty()) {
        // TODO: handle default device from env variable
    }

    deviceENV = getEnvVar("AF_OPENCL_DEFAULT_DEVICE_TYPE");
    if (!default_device_set && !deviceENV.empty()) {
        // TODO: handle default device by type env variable
    }

    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    string noGraphicsENV = getEnvVar("AF_DISABLE_GRAPHICS");
    if (fgMngr->plugin().isLoaded() && noGraphicsENV.empty()) {
        // TODO: handle forge shared contexts
    }

    mUserDeviceOffset = mDevices.size();

    // TODO: init other needed libraries?
    // blas? program cache?
    // AF_TRACE("Default device: {}", getActiveDeviceId());
}

spdlog::logger* DeviceManager::getLogger() { return logger.get(); }

DeviceManager& DeviceManager::getInstance() {
    static auto* my_instance = new DeviceManager();
    return *my_instance;
}

void DeviceManager::setMemoryManager(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a memory manager and the default memory
    // manager still hasn't been initialized, so initialize it anyways so we
    // don't inadvertently reset to it when we first call memoryManager()
    memoryManager();
    // Calls shutdown() on the existing memory manager.
    if (memManager) { memManager->shutdownAllocator(); }
    memManager = std::move(newMgr);
    // Set the backend memory manager for this new manager to register native
    // functions correctly.
    std::unique_ptr<oneapi::Allocator> deviceMemoryManager(
        new oneapi::Allocator());
    memManager->setAllocator(std::move(deviceMemoryManager));
    memManager->initialize();
}

void DeviceManager::resetMemoryManager() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(
        new common::DefaultMemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                         AF_MEM_DEBUG || AF_ONEAPI_MEM_DEBUG));
    setMemoryManager(std::move(mgr));
}

void DeviceManager::setMemoryManagerPinned(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a pinned memory manager and the default
    // memory manager still hasn't been initialized, so initialize it anyways so
    // we don't inadvertently reset to it when we first call
    // pinnedMemoryManager()
    pinnedMemoryManager();
    // Calls shutdown() on the existing memory manager.
    if (pinnedMemManager) { pinnedMemManager->shutdownAllocator(); }
    // Set the backend pinned memory manager for this new manager to register
    // native functions correctly.
    pinnedMemManager = std::move(newMgr);
    std::unique_ptr<oneapi::AllocatorPinned> deviceMemoryManager(
        new oneapi::AllocatorPinned());
    pinnedMemManager->setAllocator(std::move(deviceMemoryManager));
    pinnedMemManager->initialize();
}

void DeviceManager::resetMemoryManagerPinned() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(
        new common::DefaultMemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                         AF_MEM_DEBUG || AF_ONEAPI_MEM_DEBUG));
    setMemoryManagerPinned(std::move(mgr));
}

DeviceManager::~DeviceManager() {
    for (int i = 0; i < getDeviceCount(); ++i) { gfxManagers[i] = nullptr; }
    memManager       = nullptr;
    pinnedMemManager = nullptr;

    // TODO: cleanup mQueues, mContexts, mDevices??
}

void DeviceManager::markDeviceForInterop(const int device,
                                         const void* wHandle) {
    ONEAPI_NOT_SUPPORTED("");
}

}  // namespace oneapi
