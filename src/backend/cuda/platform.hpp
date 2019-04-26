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
#include <cu_handles.hpp>
#include <string>

namespace spdlog {
class logger;
}

namespace graphics {
class ForgeManager;
}

namespace cuda {

class GraphicsResourceManager;
class MemoryManager;
class MemoryManagerPinned;
class PlanCache;

int getBackend();

std::string getDeviceInfo();
std::string getDeviceInfo(int device);

std::string getPlatformInfo();

std::string int_version_to_string(int version);
std::string getDriverVersion();

std::string getCUDARuntimeVersion();

bool isDoubleSupported(int device);

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

bool& evalFlag();

MemoryManager& memoryManager();

MemoryManagerPinned& pinnedMemoryManager();

graphics::ForgeManager& forgeManager();

GraphicsResourceManager& interopManager();

PlanCache& fftManager();

BlasHandle blasHandle();

SolveHandle solverDnHandle();

SparseHandle sparseHandle();

}  // namespace cuda
