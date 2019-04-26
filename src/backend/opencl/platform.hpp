/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <CL/cl2.hpp>
#pragma GCC diagnostic pop

#include <af/opencl.h>
#include <string>

namespace boost {
template<typename T>
class shared_ptr;

namespace compute {
class program_cache;
}
}  // namespace boost

namespace graphics {
class ForgeManager;
}

namespace opencl {

// Forward declarations
class GraphicsResourceManager;
struct kc_entry_t;  // kernel cache entry
class MemoryManager;
class MemoryManagerPinned;
class PlanCache;  // clfft

static inline bool verify_present(std::string pname, const char* ref) {
    return pname.find(ref) != std::string::npos;
}

int getBackend();

std::string getDeviceInfo();

int getDeviceCount();

int getActiveDeviceId();

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

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

std::string getPlatformName(const cl::Device& device);

int setDevice(int device);

void addDeviceContext(cl_device_id dev, cl_context cxt, cl_command_queue que);

void setDeviceContext(cl_device_id dev, cl_context cxt);

void removeDeviceContext(cl_device_id dev, cl_context ctx);

void sync(int device);

bool synchronize_calls();

int getActiveDeviceType();

int getActivePlatform();

bool& evalFlag();

MemoryManager& memoryManager();

MemoryManagerPinned& pinnedMemoryManager();

graphics::ForgeManager& forgeManager();

GraphicsResourceManager& interopManager();

PlanCache& fftManager();

void addKernelToCache(int device, const std::string& key,
                      const kc_entry_t entry);

void removeKernelFromCache(int device, const std::string& key);

kc_entry_t kernelCache(int device, const std::string& key);

static afcl::platform getPlatformEnum(cl::Device dev) {
    std::string pname = getPlatformName(dev);
    if (verify_present(pname, "AMD")) return AFCL_PLATFORM_AMD;
    if (verify_present(pname, "NVIDIA")) return AFCL_PLATFORM_NVIDIA;
    if (verify_present(pname, "INTEL")) return AFCL_PLATFORM_INTEL;
    if (verify_present(pname, "APPLE")) return AFCL_PLATFORM_APPLE;
    if (verify_present(pname, "BEIGNET")) return AFCL_PLATFORM_BEIGNET;
    if (verify_present(pname, "POCL")) return AFCL_PLATFORM_POCL;
    return AFCL_PLATFORM_UNKNOWN;
}

void setActiveContext(int device);

}  // namespace opencl
