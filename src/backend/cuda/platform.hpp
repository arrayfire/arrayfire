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

#ifdef WITH_CUDNN
struct cudnnContext;
typedef struct cudnnContext* cudnnHandle_t;
#endif

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
namespace cuda {

class GraphicsResourceManager;
class PlanCache;

int getBackend();

std::string getDeviceInfo() noexcept;
std::string getDeviceInfo(int device) noexcept;

std::string getPlatformInfo() noexcept;

std::string getDriverVersion() noexcept;

// Returns the cuda runtime version as a string for the current build. If no
// runtime is found or an error occured, the string "N/A" is returned
std::string getCUDARuntimeVersion() noexcept;

// Returns true if double is supported by the device
bool isDoubleSupported(int device) noexcept;

// Returns true if half is supported by the device
bool isHalfSupported(int device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

int& getMaxJitSize();

int getDeviceCount();

void init();

int getActiveDeviceId();

int getDeviceNativeId(int device);

cudaStream_t getStream(int device);

cudaStream_t getActiveStream();

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

size_t getL2CacheSize(const int device);

// Returns int[3] of maxGridSize
const int* getMaxGridSize(const int device);

unsigned getMemoryBusWidth(const int device);

// maximum nr of threads the device really can run in parallel, without
// scheduling
unsigned getMaxParallelThreads(const int device);

unsigned getMultiProcessorCount(const int device);

int setDevice(int device);

void sync(int device);

// Returns true if the AF_SYNCHRONIZE_CALLS environment variable is set to 1
bool synchronize_calls();

const cudaDeviceProp& getDeviceProp(const int device);

std::pair<int, int> getComputeCapability(const int device);

bool& evalFlag();

MemoryManagerBase& memoryManager();

MemoryManagerBase& pinnedMemoryManager();

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManager();

void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManagerPinned();

arrayfire::common::ForgeManager& forgeManager();

GraphicsResourceManager& interopManager();

PlanCache& fftManager();

BlasHandle blasHandle();

#ifdef WITH_CUDNN
cudnnHandle_t nnHandle();
#endif

SolveHandle solverDnHandle();

SparseHandle sparseHandle();

}  // namespace cuda
}  // namespace arrayfire
