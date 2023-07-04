/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/oneapi.h>

#include <sycl/sycl.hpp>

#include <memory>
#include <string>

// Forward declarations
namespace spdlog {
class logger;
}

namespace arrayfire {
namespace common {
class MemoryManagerBase;
class ForgeManager;
}  // namespace common
}  // namespace arrayfire

using arrayfire::common::MemoryManagerBase;

namespace arrayfire {
namespace oneapi {

// Forward declarations
class GraphicsResourceManager;
class PlanCache;  // clfft

bool verify_present(const std::string& pname, const std::string ref);

int getBackend();

std::string getDeviceInfo() noexcept;

int getDeviceCount() noexcept;

void init();

unsigned getActiveDeviceId();

int& getMaxJitSize();

const sycl::context& getContext();

sycl::queue& getQueue();

/// Return a handle to the queue for the device.
///
/// \param[in] device The device of the returned queue
/// \returns The handle to the queue
sycl::queue* getQueueHandle(int device);

const sycl::device& getDevice(int id = -1);

const std::string& getActiveDeviceBaseBuildFlags();

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

unsigned getMemoryBusWidth(const sycl::device& device);

size_t getL2CacheSize(const sycl::device& device);

unsigned getComputeUnits(const sycl::device& device);

// maximum nr of threads the device really can run in parallel, without
// scheduling
unsigned getMaxParallelThreads(const sycl::device& device);

// sycl::device::is_cpu,is_gpu,is_accelerator
sycl::info::device_type getDeviceType();

bool isHostUnifiedMemory(const sycl::device& device);

bool OneAPICPUOffload(bool forceOffloadOSX = true);

bool isGLSharingSupported();

bool isDoubleSupported(unsigned device);

// Returns true if 16-bit precision floats are supported by the device
bool isHalfSupported(unsigned device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

std::string getPlatformName(const sycl::device& device);

int setDevice(int device);

void addDeviceContext(sycl::device& dev, sycl::context& ctx, sycl::queue& que);

void setDeviceContext(sycl::device& dev, sycl::context& ctx);

void removeDeviceContext(sycl::device& dev, sycl::context& ctx);

void sync(int device);

bool synchronize_calls();

int getActiveDeviceType();

int getActivePlatform();

bool& evalFlag();

MemoryManagerBase& memoryManager();

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManager();

MemoryManagerBase& pinnedMemoryManager();

void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManagerPinned();

arrayfire::common::ForgeManager& forgeManager();

GraphicsResourceManager& interopManager();

PlanCache& fftManager();

// afcl::platform getPlatformEnum(cl::Device dev);

void setActiveContext(int device);

}  // namespace oneapi
}  // namespace arrayfire
