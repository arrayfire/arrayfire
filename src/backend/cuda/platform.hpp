/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>

/* Forward declarations of Opaque structure holding
 * the following library contexts
 *  * cuBLAS
 *  * cuSparse
 *  * cuSolver
 */
struct cublasContext;
typedef struct cublasContext* BlasHandle;
struct cusparseContext;
typedef struct cusparseContext* SparseHandle;
struct cusolverDnContext;
typedef struct cusolverDnContext* SolveHandle;
struct cudnnContext;
typedef struct cudnnContext* cudnnHandle_t;

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

namespace cuda {

class GraphicsResourceManager;
class PlanCache;

int getBackend();

std::string getDeviceInfo() noexcept;
std::string getDeviceInfo(int device) noexcept;

std::string getPlatformInfo() noexcept;

std::string getDriverVersion();

// Returns the cuda runtime version as a string for the current build. If no
// runtime is found or an error occured, the string "N/A" is returned
std::string getCUDARuntimeVersion() noexcept;

// Returns true if double is supported by the device
bool isDoubleSupported(int device);

// Returns true if half is supported by the device
bool isHalfSupported(int device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

unsigned getMaxJitSize();

int getDeviceCount();

int getActiveDeviceId();

int getDeviceNativeId(int device);

cudaStream_t getStream(int device);

cudaStream_t getActiveStream();

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

int setDevice(int device);

void sync(int device);

// Returns true if the AF_SYNCHRONIZE_CALLS environment variable is set to 1
bool synchronize_calls();

cudaDeviceProp getDeviceProp(int device);

std::pair<int, int> getComputeCapability(const int device);

bool &evalFlag();

MemoryManagerBase& memoryManager();

MemoryManagerBase& pinnedMemoryManager();

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManager();

void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManagerPinned();

graphics::ForgeManager& forgeManager();

GraphicsResourceManager &interopManager();

PlanCache &fftManager();

BlasHandle blasHandle();

cudnnHandle_t nnHandle();

SolveHandle solverDnHandle();

SparseHandle sparseHandle();

}  // namespace cuda
