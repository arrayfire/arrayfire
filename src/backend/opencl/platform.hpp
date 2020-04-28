/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wcatch-value="
#endif
#include <CL/cl2.hpp>
#pragma GCC diagnostic pop

#include <af/opencl.h>
#include <memory>
#include <string>

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
}  // namespace common

using common::memory::MemoryManagerBase;

namespace opencl {

// Forward declarations
class GraphicsResourceManager;
struct kc_entry_t;  // kernel cache entry
class PlanCache;    // clfft

static inline bool verify_present(std::string pname, const char* ref) {
    return pname.find(ref) != std::string::npos;
}

int getBackend();

std::string getDeviceInfo() noexcept;

int getDeviceCount() noexcept;

unsigned getActiveDeviceId();

unsigned getMaxJitSize();

const cl::Context& getContext();

cl::CommandQueue& getQueue();

const cl::Device& getDevice(int id = -1);

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

cl_device_type getDeviceType();

bool isHostUnifiedMemory(const cl::Device& device);

bool OpenCLCPUOffload(bool forceOffloadOSX = true);

bool isGLSharingSupported();

bool isDoubleSupported(int device);

// Returns true if 16-bit precision floats are supported by the device
bool isHalfSupported(int device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

std::string getPlatformName(const cl::Device& device);

int setDevice(int device);

void addDeviceContext(cl_device_id dev, cl_context ctx, cl_command_queue que);

void setDeviceContext(cl_device_id dev, cl_context ctx);

void removeDeviceContext(cl_device_id dev, cl_context ctx);

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

graphics::ForgeManager& forgeManager();

GraphicsResourceManager& interopManager();

PlanCache& fftManager();

void addKernelToCache(int device, const std::string& key,
                      const kc_entry_t entry);

void removeKernelFromCache(int device, const std::string& key);

kc_entry_t kernelCache(int device, const std::string& key);

afcl::platform getPlatformEnum(cl::Device dev);

void setActiveContext(int device);

}  // namespace opencl
