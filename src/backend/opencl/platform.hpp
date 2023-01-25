/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cl2hpp.hpp>
#include <af/opencl.h>

#include <memory>
#include <string>

// Forward declarations
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

class Version;
}  // namespace common
}  // namespace arrayfire

using arrayfire::common::MemoryManagerBase;

namespace arrayfire {
namespace opencl {

// Forward declarations
class GraphicsResourceManager;
class PlanCache;  // clfft

bool verify_present(const std::string& pname, const std::string ref);

int getBackend();

std::string getDeviceInfo() noexcept;

int getDeviceCount() noexcept;

void init();

int getActiveDeviceId();

int& getMaxJitSize();

const cl::Context& getContext();

cl::CommandQueue& getQueue();

const cl::Device& getDevice(int id = -1);

const std::string& getActiveDeviceBaseBuildFlags();

/// Returns the set of all OpenCL C Versions the device supports. The values
/// are sorted from oldest to latest.
std::vector<common::Version> getOpenCLCDeviceVersion(const cl::Device& device);

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

inline unsigned getMemoryBusWidth(const cl::Device& device) {
    return device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
}

// OCL only reports on L1 cache, so we have to estimate the L2 Cache
// size. From studying many GPU cards, it is noticed that their is a
// direct correlation between Cache line and L2 Cache size:
//      - 16KB L2 Cache for each bit in Cache line.
//        Example: RTX3070 (4096KB of L2 Cache, 256Bit of Cache
//        line)
//                   --> 256*16KB = 4096KB
//      - This is also valid for all AMD GPU's
//      - Exceptions
//          * GTX10XX series have 8KB per bit of cache line
//          * iGPU (64bit cacheline) have 5KB per bit of cache line
inline size_t getL2CacheSize(const cl::Device& device) {
    const unsigned cacheLine{getMemoryBusWidth(device)};
    return cacheLine * 1024ULL *
           (cacheLine == 64 ? 5
            : device.getInfo<CL_DEVICE_NAME>().find("GTX 10") ==
                    std::string::npos
                ? 16
                : 8);
}

inline unsigned getComputeUnits(const cl::Device& device) {
    return device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
}

// maximum nr of threads the device really can run in parallel, without
// scheduling
inline unsigned getMaxParallelThreads(const cl::Device& device) {
    return getComputeUnits(device) * 2048;
}

cl_device_type getDeviceType();

bool OpenCLCPUOffload(bool forceOffloadOSX = true);

bool isGLSharingSupported();

bool isDoubleSupported(unsigned device);
inline bool isDoubleSupported(const cl::Device& device) {
    // 64bit fp is an optional extension
    return (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp64") !=
            std::string::npos);
}

// Returns true if 16-bit precision floats are supported by the device
bool isHalfSupported(unsigned device);
inline bool isHalfSupported(const cl::Device& device) {
    // 16bit fp is an option extension
    return (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp16") !=
            std::string::npos);
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

std::string getPlatformName(const cl::Device& device);

int setDevice(int device);

void addDeviceContext(cl_device_id dev, cl_context ctx, cl_command_queue que);

void setDeviceContext(cl_device_id dev, cl_context ctx);

void removeDeviceContext(cl_device_id dev, cl_context ctx);

void sync(int device);

bool synchronize_calls();

int getActiveDeviceType();

cl::Platform& getActivePlatform();

afcl::platform getActivePlatformVendor();

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

afcl::platform getPlatformEnum(cl::Device dev);

void setActiveContext(int device);

}  // namespace opencl
}  // namespace arrayfire
