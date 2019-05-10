/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(OS_WIN)
#include <windows.h>
#endif

#include <GraphicsResourceManager.hpp>
#include <common/Logger.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <cublas.hpp>
#include <cufft.hpp>
#include <cusolverDn.hpp>
#include <cusparse.hpp>
#include <driver.h>
#include <device_manager.hpp>
#include <err_cuda.hpp>
#include <memory.hpp>
#include <spdlog/spdlog.h>
#include <version.hpp>
#include <af/cuda.h>
#include <af/version.h>
// cuda_gl_interop.h does not include OpenGL headers for ARM
#include <common/graphics_common.hpp>
#define __gl_h_  // FIXME Hack to avoid gl.h inclusion by cuda_gl_interop.h
#include <cuda_gl_interop.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using std::call_once;
using std::once_flag;
using std::ostringstream;
using std::runtime_error;
using std::string;
using std::to_string;
using std::unique_ptr;

namespace cuda {

static const std::string get_system(void) {
    std::string arch = (sizeof(void *) == 4) ? "32-bit " : "64-bit ";

    return arch +
#if defined(OS_LNX)
           "Linux";
#elif defined(OS_WIN)
           "Windows";
#elif defined(OS_MAC)
           "Mac OSX";
#endif
}

static inline int getMinSupportedCompute(int cudaMajorVer) {
    // Vector of minimum supported compute versions
    // for CUDA toolkit (i+1).* where i is the index
    // of the vector
    static const std::array<int, 10> minSV{{1, 1, 1, 1, 1, 1, 2, 2, 3, 3}};

    int CVSize = static_cast<int>(minSV.size());
    return (cudaMajorVer > CVSize ? minSV[CVSize - 1]
                                  : minSV[cudaMajorVer - 1]);
}

unique_ptr<cublasHandle>& cublasManager(const int deviceId) {
    thread_local unique_ptr<cublasHandle> handles[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];

    call_once(initFlags[deviceId], [&] {
        handles[deviceId].reset(new cublasHandle());
        // TODO(pradeep) When multiple streams per device
        // is added to CUDA backend, move the cublasSetStream
        // call outside of call_once scope.
        CUBLAS_CHECK(cublasSetStream(*handles[deviceId],
                                     cuda::getStream(deviceId)));
    });

    return handles[deviceId];
}

unique_ptr<PlanCache>& cufftManager(const int deviceId) {
    thread_local unique_ptr<PlanCache> caches[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];
    call_once(initFlags[deviceId],
              [&] { caches[deviceId].reset(new PlanCache()); });
    return caches[deviceId];
}

unique_ptr<cusolverDnHandle>& cusolverManager(const int deviceId) {
    thread_local unique_ptr<cusolverDnHandle>
        handles[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];
    call_once(initFlags[deviceId], [&] {
        handles[deviceId].reset(new cusolverDnHandle());
        // TODO(pradeep) When multiple streams per device
        // is added to CUDA backend, move the cublasSetStream
        // call outside of call_once scope.
        CUSOLVER_CHECK(cusolverDnSetStream(*handles[deviceId],
                                           cuda::getStream(deviceId)));
    });
    // TODO(pradeep) prior to this change, stream was being synced in get solver
    // handle because of some cusolver bug. Re-enable that if this change
    // doesn't work and sovler tests fail.
    // https://gist.github.com/shehzan10/414c3d04a40e7c4a03ed3c2e1b9072e7
    // cuSolver Streams patch:
    // CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(deviceId)));

    return handles[deviceId];
}

unique_ptr<cusparseHandle>& cusparseManager(const int deviceId) {
    thread_local unique_ptr<cusparseHandle> handles[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];
    call_once(initFlags[deviceId], [&] {
        handles[deviceId].reset(new cusparseHandle());
        // TODO(pradeep) When multiple streams per device
        // is added to CUDA backend, move the cublasSetStream
        // call outside of call_once scope.
        CUSPARSE_CHECK(cusparseSetStream(*handles[deviceId],
                                         cuda::getStream(deviceId)));
    });
    return handles[deviceId];
}

DeviceManager::~DeviceManager() {
    // Reset unique_ptrs for all cu[BLAS | Sparse | Solver]
    // handles of all devices
    for (int i = 0; i < nDevices; ++i) {
        setDevice(i);
        cublasManager(i).reset();
        cufftManager(i).reset();
        cusolverManager(i).reset();
        cusparseManager(i).reset();
    }
}

int getBackend() { return AF_BACKEND_CUDA; }

string getDeviceInfo(int device) {
    cudaDeviceProp dev = getDeviceProp(device);

    size_t mem_gpu_total = dev.totalGlobalMem;
    // double cc = double(dev.major) + double(dev.minor) / 10;

    bool show_braces = getActiveDeviceId() == device;

    string id = (show_braces ? string("[") : "-") + to_string(device) +
                (show_braces ? string("]") : "-");
    string name(dev.name);
    string memory = to_string((mem_gpu_total / (1024 * 1024)) +
                              !!(mem_gpu_total % (1024 * 1024))) +
                    string(" MB");
    string compute = string("CUDA Compute ") + to_string(dev.major) +
                     string(".") + to_string(dev.minor);

    string info = id + string(" ") + name + string(", ") + memory +
                  string(", ") + compute + string("\n");
    return info;
}

string getDeviceInfo() {
    ostringstream info;
    info << "ArrayFire v" << AF_VERSION << " (CUDA, " << get_system()
         << ", build " << AF_REVISION << ")" << std::endl;
    info << getPlatformInfo();
    for (int i = 0; i < getDeviceCount(); ++i) { info << getDeviceInfo(i); }
    return info.str();
}

string getPlatformInfo() {
    string driverVersion    = getDriverVersion();
    std::string cudaRuntime = getCUDARuntimeVersion();
    string platform         = "Platform: CUDA Toolkit " + cudaRuntime;
    if (!driverVersion.empty()) {
        platform.append(", Driver: ");
        platform.append(driverVersion);
    }
    platform.append("\n");
    return platform;
}

bool isDoubleSupported(int device) {
    UNUSED(device);
    return true;
}

void devprop(char *d_name, char *d_platform, char *d_toolkit, char *d_compute) {
    if (getDeviceCount() <= 0) { return; }

    cudaDeviceProp dev = getDeviceProp(getActiveDeviceId());

    // Name
    snprintf(d_name, 256, "%s", dev.name);

    // Platform
    std::string cudaRuntime = getCUDARuntimeVersion();
    snprintf(d_platform, 10, "CUDA");
    snprintf(d_toolkit, 64, "v%s", cudaRuntime.c_str());

    // Compute Version
    snprintf(d_compute, 10, "%d.%d", dev.major, dev.minor);

    // Sanitize input
    for (int i = 0; i < 256; i++) {
        if (d_name[i] == ' ') {
            if (d_name[i + 1] == 0 || d_name[i + 1] == ' ')
                d_name[i] = 0;
            else
                d_name[i] = '_';
        }
    }
}

string getDriverVersion() {
    char driverVersion[1024] = {
        " ",
    };
    int x = nvDriverVersion(driverVersion, sizeof(driverVersion));
    if (x != 1) {
// Windows, OSX, Tegra Need a new way to fetch driver
#if !defined(OS_WIN) && !defined(OS_MAC) && !defined(__arm__) && \
    !defined(__aarch64__)
        throw runtime_error("Invalid driver");
#endif
        int driver = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        return to_string(driver);
    } else {
        return string(driverVersion);
    }
}

string int_version_to_string(int version) {
    return to_string(version / 1000) + "." +
           to_string((int)((version % 1000) / 10.));
}

string getCUDARuntimeVersion() {
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    return int_version_to_string(runtime);
}

unsigned getMaxJitSize() {
    const int MAX_JIT_LEN = 100;

    thread_local int length = 0;
    if (length == 0) {
        std::string env_var = getEnvVar("AF_CUDA_MAX_JIT_LEN");
        if (!env_var.empty()) {
            length = std::stoi(env_var);
        } else {
            length = MAX_JIT_LEN;
        }
    }

    return length;
}

int& tlocalActiveDeviceId() {
    thread_local int activeDeviceId = 0;

    return activeDeviceId;
}

int getDeviceCount() { return DeviceManager::getInstance().nDevices; }

int getActiveDeviceId() { return tlocalActiveDeviceId(); }

int getDeviceNativeId(int device) {
    if (device < (int)DeviceManager::getInstance().cuDevices.size())
        return DeviceManager::getInstance().cuDevices[device].nativeId;
    return -1;
}

int getDeviceIdFromNativeId(int nativeId) {
    DeviceManager &mngr = DeviceManager::getInstance();

    int devId = 0;
    for (devId = 0; devId < mngr.nDevices; ++devId) {
        if (nativeId == mngr.cuDevices[devId].nativeId) break;
    }
    return devId;
}

cudaStream_t getStream(int device) {
    static std::once_flag streamInitFlags[DeviceManager::MAX_DEVICES];

    std::call_once(streamInitFlags[device], [device]() {
        DeviceManager &inst = DeviceManager::getInstance();
        CUDA_CHECK(cudaStreamCreate(&(inst.streams[device])));
    });

    return DeviceManager::getInstance().streams[device];
}

cudaStream_t getActiveStream() { return getStream(getActiveDeviceId()); }

size_t getDeviceMemorySize(int device) {
    return getDeviceProp(device).totalGlobalMem;
}

size_t getHostMemorySize() { return common::getHostMemorySize(); }

int setDevice(int device) {
    return DeviceManager::getInstance().setActiveDevice(device);
}

cudaDeviceProp getDeviceProp(int device) {
    if (device < (int)DeviceManager::getInstance().cuDevices.size())
        return DeviceManager::getInstance().cuDevices[device].prop;
    return DeviceManager::getInstance().cuDevices[0].prop;
}

MemoryManager &memoryManager() {
    static std::once_flag flag;

    DeviceManager &inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() { inst.memManager.reset(new MemoryManager()); });

    return *(inst.memManager.get());
}

MemoryManagerPinned &pinnedMemoryManager() {
    static std::once_flag flag;

    DeviceManager &inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() {
        inst.pinnedMemManager.reset(new MemoryManagerPinned());
    });

    return *(inst.pinnedMemManager.get());
}

graphics::ForgeManager &forgeManager() {
    return *(DeviceManager::getInstance().fgMngr);
}

GraphicsResourceManager &interopManager() {
    static std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = getActiveDeviceId();

    DeviceManager &inst = DeviceManager::getInstance();

    std::call_once(initFlags[id], [&] {
        inst.gfxManagers[id].reset(new GraphicsResourceManager());
    });

    return *(inst.gfxManagers[id].get());
}

PlanCache &fftManager() {
    return *(cufftManager(cuda::getActiveDeviceId()).get());
}

BlasHandle blasHandle() {
    return *cublasManager(cuda::getActiveDeviceId());
}

SolveHandle solverDnHandle() {
    return *cusolverManager(cuda::getActiveDeviceId());
}

SparseHandle sparseHandle() {
    return *cusparseManager(cuda::getActiveDeviceId());
}

void sync(int device) {
    int currDevice = getActiveDeviceId();
    setDevice(device);
    CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));
    setDevice(currDevice);
}

bool synchronize_calls() {
    static const bool sync = getEnvVar("AF_SYNCHRONOUS_CALLS") == "1";
    return sync;
}

bool &evalFlag() {
    thread_local bool flag = true;
    return flag;
}

}  // namespace cuda

af_err afcu_get_stream(cudaStream_t *stream, int id) {
    try {
        *stream = cuda::getStream(id);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcu_get_native_id(int *nativeid, int id) {
    try {
        *nativeid = cuda::getDeviceNativeId(id);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcu_set_native_id(int nativeid) {
    try {
        cuda::setDevice(cuda::getDeviceIdFromNativeId(nativeid));
    }
    CATCHALL;
    return AF_SUCCESS;
}
