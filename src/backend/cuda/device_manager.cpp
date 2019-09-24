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
#include <cublas_v2.h>  // needed for af/cuda.h
#include <device_manager.hpp>
#include <driver.h>
#include <err_cuda.hpp>
#include <memory.hpp>
#include <platform.hpp>
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

using std::begin;
using std::end;
using std::find_if;
using std::make_pair;
using std::pair;
using std::string;
using std::stringstream;

namespace cuda {

void findJitDevCompute(pair<int, int> &prop) {
    struct cuNVRTCcompute {
        /// The CUDA Toolkit version returned by cudaRuntimeGetVersion
        int cuda_version;
        /// Maximum major compute flag supported by cuda_version
        int major;
        /// Maximum minor compute flag supported by cuda_version
        int minor;
    };
    static const cuNVRTCcompute Toolkit2Compute[] = {
        {10010, 7, 5}, {10000, 7, 2}, {9020, 7, 2}, {9010, 7, 2},
        {9000, 7, 2},  {8000, 5, 3},  {7050, 5, 3}, {7000, 5, 3}};
    int runtime_cuda_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_cuda_ver));
    auto tkit_max_compute =
        find_if(begin(Toolkit2Compute), end(Toolkit2Compute),
                [runtime_cuda_ver](cuNVRTCcompute v) {
                    return runtime_cuda_ver == v.cuda_version;
                });
    if ((tkit_max_compute == end(Toolkit2Compute)) ||
        (prop.first > tkit_max_compute->major &&
         prop.second > tkit_max_compute->minor)) {
        prop = make_pair(tkit_max_compute->major, tkit_max_compute->minor);
    }
}

pair<int, int> getComputeCapability(const int device) {
    return DeviceManager::getInstance().devJitComputes[device];
}

// pulled from CUTIL from CUDA SDK
static inline int compute2cores(int major, int minor) {
    struct {
        int compute;  // 0xMm (hex), M = major version, m = minor version
        int cores;
    } gpus[] = {
        {0x10, 8},   {0x11, 8},   {0x12, 8},   {0x13, 8},   {0x20, 32},
        {0x21, 48},  {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
        {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128},
        {0x62, 128}, {0x70, 64},  {0x75, 64},  {-1, -1},
    };

    for (int i = 0; gpus[i].compute != -1; ++i) {
        if (gpus[i].compute == (major << 4) + minor) return gpus[i].cores;
    }
    return 0;
}

// Return true if greater, false if lesser.
// if equal, it continues to next comparison
#define COMPARE(a, b, f)                   \
    do {                                   \
        if ((a)->f > (b)->f) return true;  \
        if ((a)->f < (b)->f) return false; \
        break;                             \
    } while (0)

static inline bool card_compare_compute(const cudaDevice_t &l,
                                        const cudaDevice_t &r) {
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, prop.major);
    COMPARE(lc, rc, prop.minor);
    COMPARE(lc, rc, flops);
    COMPARE(lc, rc, prop.totalGlobalMem);
    COMPARE(lc, rc, nativeId);
    return false;
}

static inline bool card_compare_flops(const cudaDevice_t &l,
                                      const cudaDevice_t &r) {
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, flops);
    COMPARE(lc, rc, prop.totalGlobalMem);
    COMPARE(lc, rc, prop.major);
    COMPARE(lc, rc, prop.minor);
    COMPARE(lc, rc, nativeId);
    return false;
}

static inline bool card_compare_mem(const cudaDevice_t &l,
                                    const cudaDevice_t &r) {
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, prop.totalGlobalMem);
    COMPARE(lc, rc, flops);
    COMPARE(lc, rc, prop.major);
    COMPARE(lc, rc, prop.minor);
    COMPARE(lc, rc, nativeId);
    return false;
}

static inline bool card_compare_num(const cudaDevice_t &l,
                                    const cudaDevice_t &r) {
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, nativeId);
    return false;
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

bool DeviceManager::checkGraphicsInteropCapability() {
    static std::once_flag checkInteropFlag;
    thread_local bool capable = true;

    std::call_once(checkInteropFlag, []() {
        unsigned int pCudaEnabledDeviceCount = 0;
        int pCudaGraphicsEnabledDeviceIds    = 0;
        cudaGetLastError();  // Reset Errors
        cudaError_t err = cudaGLGetDevices(
            &pCudaEnabledDeviceCount, &pCudaGraphicsEnabledDeviceIds,
            getDeviceCount(), cudaGLDeviceListAll);
        if (err == cudaErrorOperatingSystem) {
            // OS Support Failure - Happens when devices are in TCC mode or
            // do not have a display connected
            capable = false;
        }
        cudaGetLastError();  // Reset Errors
    });

    return capable;
}

DeviceManager &DeviceManager::getInstance() {
    static DeviceManager *my_instance = new DeviceManager();
    return *my_instance;
}

void DeviceManager::setMemoryManager(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    // If an existing memory manager exists, shutdown()
    if (memManager) { memManager->shutdown(); }
    // Set the backend memory manager for this new manager to register native
    // functions correctly
    std::unique_ptr<cuda::NativeMemoryInterface> deviceMemoryManager;
    deviceMemoryManager.reset(new cuda::NativeMemoryInterface());
    newMgr->setNativeMemoryInterface(std::move(deviceMemoryManager));
    newMgr->initialize();
    memManager = std::move(newMgr);
}

void DeviceManager::resetMemoryManager() {
    std::unique_ptr<MemoryManagerBase> mgr;
    mgr.reset(new common::MemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                        AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG));
    std::unique_ptr<cuda::NativeMemoryInterface> deviceMemoryManager;
    deviceMemoryManager.reset(new cuda::NativeMemoryInterface());
    mgr->setNativeMemoryInterface(std::move(deviceMemoryManager));
    mgr->initialize();

    setMemoryManager(std::move(mgr));
}

void DeviceManager::setMemoryManagerPinned(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    // If an existing memory manager exists, shutdown()
    if (pinnedMemManager) { pinnedMemManager->shutdown(); }
    // Set the backend memory manager for this new manager to register native
    // functions correctly
    std::unique_ptr<cuda::NativeMemoryInterfacePinned> deviceMemoryManager;
    deviceMemoryManager.reset(new cuda::NativeMemoryInterfacePinned());
    newMgr->setNativeMemoryInterface(std::move(deviceMemoryManager));
    newMgr->initialize();
    pinnedMemManager = std::move(newMgr);
}

void DeviceManager::resetMemoryManagerPinned() {
    std::unique_ptr<MemoryManagerBase> mgr;
    mgr.reset(new common::MemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                        AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG));
    std::unique_ptr<cuda::NativeMemoryInterfacePinned> deviceMemoryManager;
    deviceMemoryManager.reset(new cuda::NativeMemoryInterfacePinned());
    mgr->setNativeMemoryInterface(std::move(deviceMemoryManager));
    mgr->initialize();

    setMemoryManagerPinned(std::move(mgr));
}

/// Struct represents the cuda toolkit version and its associated minimum
/// required driver versions.
struct ToolkitDriverVersions {
    /// The CUDA Toolkit version returned by cudaDriverGetVersion or
    /// cudaRuntimeGetVersion
    int version;

    /// The minimum GPU driver version required for the \p version toolkit on
    /// Linux or macOS
    float unix_min_version;

    /// The minimum GPU driver version required for the \p version toolkit on
    /// Windows
    float windows_min_version;
};

/// Map giving the minimum device driver needed in order to run a given version
/// of CUDA for both Linux/Mac and Windows from:
/// https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
// clang-format off
static const ToolkitDriverVersions
    CudaToDriverVersion[] = {
        {10010, 418.39f, 418.96f},
        {10000, 410.48f, 411.31f},
        {9020,  396.37f, 398.26f},
        {9010,  390.46f, 391.29f},
        {9000,  384.81f, 385.54f},
        {8000,  375.26f, 376.51f},
        {7050,  352.31f, 353.66f},
        {7000,  346.46f, 347.62f}};
// clang-format on

/// A debug only function that checks to see if the driver or runtime
/// function is part of the CudaToDriverVersion array. If the runtime
/// version is not part of the array then an error is thrown in debug
/// mode. If the driver version is not part of the array, then a message
/// is displayed in the error stream.
///
/// \param[in] runtime_version  The version integer returned by
///                             cudaRuntimeGetVersion
/// \param[in] driver_version   The version integer returned by
///                             cudaDriverGetVersion
/// \note: only works in debug builds
void debugRuntimeCheck(int runtime_version, int driver_version) {
#ifndef NDEBUG
    auto runtime_it =
        find_if(begin(CudaToDriverVersion), end(CudaToDriverVersion),
                [runtime_version](ToolkitDriverVersions ver) {
                    return runtime_version == ver.version;
                });
    auto driver_it =
        find_if(begin(CudaToDriverVersion), end(CudaToDriverVersion),
                [driver_version](ToolkitDriverVersions ver) {
                    return driver_version == ver.version;
                });

    // If the runtime version is not part of the CudaToDriverVersion array,
    // display a message in the trace. Do not throw an error unless this is
    // a debug build
    if (runtime_it == end(CudaToDriverVersion)) {
        char buf[1024];
        char err_msg[] =
            "WARNING: CUDA runtime version(%s) not recognized. Please "
            "create an issue or a pull request on the ArrayFire repository to "
            "update the CudaToDriverVersion variable with this version of "
            "the CUDA Toolkit.\n";
        snprintf(buf, 1024, err_msg,
                 int_version_to_string(runtime_version).c_str());
        fprintf(stderr, err_msg,
                int_version_to_string(runtime_version).c_str());
        AF_ERROR(buf, AF_ERR_RUNTIME);
    }

    if (driver_it == end(CudaToDriverVersion)) {
        char err_msg[] =
            "WARNING: CUDA driver version(%s) not part of the "
            "CudaToDriverVersion array. Please create an issue or a pull "
            "request on the ArrayFire repository to update the "
            "CudaToDriverVersion variable with this version of the CUDA "
            "Toolkit.\n";
        fprintf(stderr, err_msg, int_version_to_string(driver_version).c_str());
    }
#endif
}

// Check if the device driver version is recent enough to run the cuda libs
// linked with afcuda:
void DeviceManager::checkCudaVsDriverVersion() {
    const std::string driverVersionString = getDriverVersion();

    int driver  = 0;
    int runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

    AF_TRACE("CUDA supported by the GPU Driver {} ArrayFire CUDA Runtime {}",
             int_version_to_string(driver), int_version_to_string(runtime));

    debugRuntimeCheck(runtime, driver);

    if (runtime > driver) {
        string msg =
            "ArrayFire was built with CUDA %s which requires GPU driver "
            "version %.2f or later. Please download and install the latest "
            "drivers from https://www.nvidia.com/drivers for your GPU. "
            "Alternatively, you could rebuild ArrayFire with CUDA Toolkit "
            "version %s to use the current drivers.";

        auto runtime_it =
            find_if(begin(CudaToDriverVersion), end(CudaToDriverVersion),
                    [runtime](ToolkitDriverVersions ver) {
                        return runtime == ver.version;
                    });

        // If the runtime version is not part of the CudaToDriverVersion
        // array, display a message in the trace. Do not throw an error
        // unless this is a debug build
        if (runtime_it == end(CudaToDriverVersion)) {
            char buf[1024];
            char err_msg[] =
                "CUDA runtime version(%s) not recognized. Please create an "
                "issue or a pull request on the ArrayFire repository to "
                "update the CudaToDriverVersion variable with this "
                "version of the CUDA Toolkit.";
            snprintf(buf, 1024, err_msg,
                     int_version_to_string(runtime).c_str());
            AF_TRACE("{}", buf);
            return;
        }

        float minimumDriverVersion =
#ifdef OS_WIN
            runtime_it->windows_min_version;
#else
            runtime_it->unix_min_version;
#endif

        char buf[1024];
        snprintf(buf, 1024, msg.c_str(), int_version_to_string(runtime).c_str(),
                 minimumDriverVersion, int_version_to_string(driver).c_str());

        AF_ERROR(buf, AF_ERR_DRIVER);
    }
}

DeviceManager::DeviceManager()
    : logger(common::loggerFactory("platform"))
    , cuDevices(0)
    , nDevices(0)
    , fgMngr(new graphics::ForgeManager()) {
    checkCudaVsDriverVersion();

    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    AF_TRACE("Found {} CUDA devices", nDevices);
    if (nDevices == 0) {
        AF_ERROR("No CUDA capable devices found", AF_ERR_DRIVER);
    }
    cuDevices.reserve(nDevices);

    int cudaRtVer = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cudaRtVer));
    int cudaMajorVer = cudaRtVer / 1000;

    for (int i = 0; i < nDevices; i++) {
        cudaDevice_t dev;
        CUDA_CHECK(cudaGetDeviceProperties(&dev.prop, i));
        if (dev.prop.major < getMinSupportedCompute(cudaMajorVer)) {
            AF_TRACE("Unsuppored device: {}", dev.prop.name);
            continue;
        } else {
            dev.flops = static_cast<size_t>(dev.prop.multiProcessorCount) *
                        compute2cores(dev.prop.major, dev.prop.minor) *
                        dev.prop.clockRate;
            dev.nativeId = i;
            AF_TRACE(
                "Found device: {} ({:0.3} GB | ~{} GFLOPs | {} SMs)",
                dev.prop.name, dev.prop.totalGlobalMem / 1024. / 1024. / 1024.,
                dev.flops / 1024. / 1024. * 2, dev.prop.multiProcessorCount);
            cuDevices.push_back(dev);
        }
    }
    nDevices = cuDevices.size();

    sortDevices();

    // Initialize all streams to 0.
    // Streams will be created in setActiveDevice()
    for (size_t i = 0; i < MAX_DEVICES; i++) {
        streams[i] = (cudaStream_t)0;
        if (i < nDevices) {
            auto prop =
                make_pair(cuDevices[i].prop.major, cuDevices[i].prop.minor);
            findJitDevCompute(prop);
            devJitComputes.emplace_back(prop);
        }
    }

    std::string deviceENV = getEnvVar("AF_CUDA_DEFAULT_DEVICE");
    AF_TRACE("AF_CUDA_DEFAULT_DEVICE: {}", deviceENV);
    if (deviceENV.empty()) {
        setActiveDevice(0, cuDevices[0].nativeId);
    } else {
        stringstream s(deviceENV);
        int def_device = -1;
        s >> def_device;
        if (def_device < 0 || def_device >= nDevices) {
            getLogger()->warn(
                "AF_CUDA_DEFAULT_DEVICE({}) out of range. Setting default "
                "device to 0.",
                def_device);
            setActiveDevice(0, cuDevices[0].nativeId);
        } else {
            setActiveDevice(def_device, cuDevices[def_device].nativeId);
        }
    }
    AF_TRACE("Default device: {}", getActiveDeviceId());
}

spdlog::logger *DeviceManager::getLogger() { return logger.get(); }

void DeviceManager::sortDevices(sort_mode mode) {
    switch (mode) {
        case memory:
            std::stable_sort(cuDevices.begin(), cuDevices.end(),
                             card_compare_mem);
            break;
        case flops:
            std::stable_sort(cuDevices.begin(), cuDevices.end(),
                             card_compare_flops);
            break;
        case compute:
            std::stable_sort(cuDevices.begin(), cuDevices.end(),
                             card_compare_compute);
            break;
        case none:
        default:
            std::stable_sort(cuDevices.begin(), cuDevices.end(),
                             card_compare_num);
            break;
    }
}

int DeviceManager::setActiveDevice(int device, int nId) {
    thread_local bool retryFlag = true;

    int numDevices = cuDevices.size();

    if (device >= numDevices) return -1;

    int old = getActiveDeviceId();

    if (nId == -1) nId = getDeviceNativeId(device);

    cudaError_t err = cudaSetDevice(nId);

    if (err == cudaSuccess) {
        tlocalActiveDeviceId() = device;
        return old;
    }

    // For the first time a thread calls setDevice,
    // if the requested device is unavailable, try checking
    // for other available devices - while loop below
    if (!retryFlag) {
        CUDA_CHECK(err);
        return old;
    }

    // Comes only when retryFlag is true. Set it to false
    retryFlag = false;

    while (true) {
        // Check for errors other than DevicesUnavailable
        // If success, return. Else throw error
        // If DevicesUnavailable, try other devices (while loop below)
        if (err != cudaErrorDeviceAlreadyInUse) {
            CUDA_CHECK(err);
            tlocalActiveDeviceId() = device;
            return old;
        }
        cudaGetLastError();  // Reset error stack
#ifndef NDEBUG
        getLogger()->warn(
            "Warning: Device {} is unavailable. Using next available "
            "device \n",
            device);
#endif
        // Comes here is the device is in exclusive mode or
        // otherwise fails streamCreate with this error.
        // All other errors will error out
        device++;
        if (device >= numDevices) break;

        // Can't call getNativeId here as it will cause an infinite loop with
        // the constructor
        nId = cuDevices[device].nativeId;

        err = cudaSetDevice(nId);
    }

    // If all devices fail with DeviceAlreadyInUse, then throw this error
    CUDA_CHECK(err);

    return old;
}

}  // namespace cuda
