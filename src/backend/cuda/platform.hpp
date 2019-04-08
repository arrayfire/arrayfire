/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <GraphicsResourceManager.hpp>
#include <cublas.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.hpp>
#include <cusolverDn.hpp>
#include <cusparse.hpp>
#include <memory.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace spdlog {
class logger;
}

namespace graphics {
class ForgeManager;
}

namespace cuda {
int getBackend();

std::string getDeviceInfo();
std::string getDeviceInfo(int device);

std::string getPlatformInfo();

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

std::pair<int, int> getComputeFlag();

struct cudaDevice_t {
    cudaDeviceProp prop;
    size_t flops;
    int nativeId;
};

bool& evalFlag();

///////////////////////// BEGIN Sub-Managers ///////////////////
//
MemoryManager& memoryManager();

MemoryManagerPinned& pinnedMemoryManager();

graphics::ForgeManager& forgeManager();

GraphicsResourceManager& interopManager();

PlanCache& fftManager();

BlasHandle blasHandle();

SolveHandle solverDnHandle();

SparseHandle sparseHandle();
//
///////////////////////// END Sub-Managers /////////////////////

class DeviceManager {
   public:
    static const unsigned MAX_DEVICES = 16;

    static bool checkGraphicsInteropCapability();

    static DeviceManager& getInstance();

    spdlog::logger* getLogger();

    friend MemoryManager& memoryManager();

    friend MemoryManagerPinned& pinnedMemoryManager();

    friend graphics::ForgeManager& forgeManager();

    friend GraphicsResourceManager& interopManager();

    friend std::string getDeviceInfo(int device);

    friend std::string getPlatformInfo();

    friend std::string getDriverVersion();

    friend std::string getCUDARuntimeVersion();

    friend std::string getDeviceInfo();

    friend int getDeviceCount();

    friend int getDeviceNativeId(int device);

    friend int getDeviceIdFromNativeId(int nativeId);

    friend cudaStream_t getStream(int device);

    friend int setDevice(int device);

    friend cudaDeviceProp getDeviceProp(int device);

    friend std::pair<int, int> getComputeFlag();

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

    void checkCudaVsDriverVersion();
    void sortDevices(sort_mode mode = flops);

    int setActiveDevice(int device, int native = -1);

    std::shared_ptr<spdlog::logger> logger;

    std::vector<cudaDevice_t> cuDevices;
    std::vector<std::pair<int, int>> devJitComputes;

    int nDevices;
    cudaStream_t streams[MAX_DEVICES];

    std::unique_ptr<graphics::ForgeManager> fgMngr;

    std::unique_ptr<MemoryManager> memManager;

    std::unique_ptr<MemoryManagerPinned> pinnedMemManager;

    std::unique_ptr<GraphicsResourceManager> gfxManagers[MAX_DEVICES];
};
}  // namespace cuda
