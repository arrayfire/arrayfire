/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <device_manager.hpp>

#if defined(OS_WIN)
#include <windows.h>
#endif

#include <GraphicsResourceManager.hpp>
#include <build_version.hpp>
#include <common/ArrayFireTypesIO.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/defines.hpp>
#include <common/graphics_common.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <cublas_v2.h>  // needed for af/cuda.h
#include <driver.h>
#include <err_cuda.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <spdlog/spdlog.h>
#include <af/cuda.h>
#include <af/version.h>
// cuda_gl_interop.h does not include OpenGL headers for ARM
// __gl_h_ should be defined by glad.h inclusion
#include <cuda_gl_interop.h>
#include <utility.hpp>

#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

using arrayfire::common::fromCudaVersion;
using arrayfire::common::getEnvVar;
using std::begin;
using std::end;
using std::find;
using std::find_if;
using std::make_pair;
using std::pair;
using std::string;
using std::stringstream;

namespace arrayfire {
namespace cuda {

struct cuNVRTCcompute {
    /// The CUDA Toolkit version returned by cudaRuntimeGetVersion
    int cudaVersion;
    /// Maximum major compute flag supported by cudaVersion
    int major;
    /// Maximum minor compute flag supported by cudaVersion
    int minor;
    /// Maximum minor compute flag supported on the embedded(Jetson) platforms
    int embedded_minor;
};

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

// clang-format off
static const int jetsonComputeCapabilities[] = {
    8070,
    7020,
    6020,
    5030,
    3020,
};
// clang-format on

// clang-format off
static const cuNVRTCcompute Toolkit2MaxCompute[] = {
    {12060, 9, 0, 0},
    {12050, 9, 0, 0},
    {12040, 9, 0, 0},
    {12030, 9, 0, 0},
    {12020, 9, 0, 0},
    {12010, 9, 0, 0},
    {12000, 9, 0, 0},
    {11080, 9, 0, 0},
    {11070, 8, 7, 0},
    {11060, 8, 6, 0},
    {11050, 8, 6, 0},
    {11040, 8, 6, 0},
    {11030, 8, 6, 0},
    {11020, 8, 6, 0},
    {11010, 8, 6, 0},
    {11000, 8, 0, 0},
    {10020, 7, 5, 2},
    {10010, 7, 5, 2},
    {10000, 7, 0, 2},
    { 9020, 7, 0, 2},
    { 9010, 7, 0, 2},
    { 9000, 7, 0, 2},
    { 8000, 5, 2, 3},
    { 7050, 5, 2, 3},
    { 7000, 5, 2, 3}};
// clang-format on

// A tuple of Compute Capability and the associated number of cores in each
// streaming multiprocessors for that architecture
struct ComputeCapabilityToStreamingProcessors {
    // The compute capability in hex
    // 0xMm (hex), M = major version, m = minor version
    int compute_capability;
    // Number of CUDA cores per SM
    int cores_per_sm;
};

/// Map giving the minimum device driver needed in order to run a given version
/// of CUDA for both Linux/Mac and Windows from:
/// https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
// clang-format off
static const ToolkitDriverVersions
    CudaToDriverVersion[] = {
        {12060, 560.28f, 560.76f},
        {12050, 555.42f, 555.85f},
        {12040, 550.54f, 551.78f},
        {12030, 545.23f, 546.12f},
        {12020, 535.104f, 537.13f},
        {12010, 530.30f, 531.14f},
        {12000, 525.85f, 528.33f},
        {11080, 520.61f, 520.06f},
        {11070, 515.48f, 516.31f},
        {11060, 510.47f, 511.65f},
        {11050, 495.29f, 496.13f},
        {11040, 470.82f, 472.50f},
        {11030, 465.19f, 465.89f},
        {11020, 460.32f, 461.33f},
        {11010, 455.32f, 456.81f},
        {11000, 450.51f, 451.82f},
        {10020, 440.33f, 441.22f},
        {10010, 418.39f, 418.96f},
        {10000, 410.48f, 411.31f},
        {9020,  396.37f, 398.26f},
        {9010,  390.46f, 391.29f},
        {9000,  384.81f, 385.54f},
        {8000,  375.26f, 376.51f},
        {7050,  352.31f, 353.66f},
        {7000,  346.46f, 347.62f}};
// clang-format on

// Vector of minimum supported compute versions for CUDA toolkit (i+1).*
// where i is the index of the vector
static const std::array<int, 12> minSV{{1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 5}};

static ComputeCapabilityToStreamingProcessors gpus[] = {
    {0x10, 8},   {0x11, 8},   {0x12, 8},   {0x13, 8},   {0x20, 32},
    {0x21, 48},  {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
    {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128},
    {0x62, 128}, {0x70, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
    {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1},
};

// pulled from CUTIL from CUDA SDK
static inline int compute2cores(unsigned major, unsigned minor) {
    for (int i = 0; gpus[i].compute_capability != -1; ++i) {
        if (static_cast<unsigned>(gpus[i].compute_capability) ==
            (major << 4U) + minor) {
            return gpus[i].cores_per_sm;
        }
    }
    return 0;
}

static inline int getMinSupportedCompute(int cudaMajorVer) {
    int CVSize = static_cast<int>(minSV.size());
    return (cudaMajorVer > CVSize ? minSV[CVSize - 1]
                                  : minSV[cudaMajorVer - 1]);
}

bool isEmbedded(pair<int, int> compute) {
    int version = compute.first * 1000 + compute.second * 10;
    return end(jetsonComputeCapabilities) !=
           find(begin(jetsonComputeCapabilities),
                end(jetsonComputeCapabilities), version);
}

bool checkDeviceWithRuntime(int runtime, pair<int, int> compute) {
    auto rt = find_if(
        begin(Toolkit2MaxCompute), end(Toolkit2MaxCompute),
        [runtime](cuNVRTCcompute c) { return c.cudaVersion == runtime; });
    if (rt == end(Toolkit2MaxCompute)) {
        spdlog::get("platform")
            ->warn(
                "CUDA runtime version({}) not recognized. Please "
                "create an issue or a pull request on the ArrayFire repository "
                "to update the Toolkit2MaxCompute array with this version of "
                "the CUDA Runtime. Continuing.",
                fromCudaVersion(runtime));
        return true;
    }

    if (rt->major >= compute.first) {
        if (rt->major == compute.first) {
            return rt->minor >= compute.second;
        } else {
            return true;
        }
    } else {
        return false;
    }
}

/// Check for compatible compute version based on runtime cuda toolkit version
void checkAndSetDevMaxCompute(pair<int, int> &computeCapability) {
    auto originalCompute = computeCapability;
    int rtCudaVer        = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rtCudaVer));
    auto tkitMaxCompute = find_if(
        begin(Toolkit2MaxCompute), end(Toolkit2MaxCompute),
        [rtCudaVer](cuNVRTCcompute v) { return rtCudaVer == v.cudaVersion; });

    bool embeddedDevice = isEmbedded(computeCapability);

    // If runtime cuda version is found in toolkit array
    // check for max possible compute for that cuda version
    if (tkitMaxCompute != end(Toolkit2MaxCompute) &&
        computeCapability.first >= tkitMaxCompute->major) {
        int minorVersion = embeddedDevice ? tkitMaxCompute->embedded_minor
                                          : tkitMaxCompute->minor;

        if (computeCapability.second > minorVersion) {
            computeCapability = make_pair(tkitMaxCompute->major, minorVersion);
            spdlog::get("platform")
                ->warn(
                    "The compute capability for the current device({}.{}) "
                    "exceeds maximum supported by ArrayFire's CUDA "
                    "runtime({}.{}). Download or rebuild the latest version of "
                    "ArrayFire to avoid this warning. Using {}.{} for JIT "
                    "compilation kernels.",
                    originalCompute.first, originalCompute.second,
                    computeCapability.first, computeCapability.second,
                    computeCapability.first, computeCapability.second);
        }
    } else if (computeCapability.first >= Toolkit2MaxCompute[0].major) {
        // If runtime cuda version is NOT found in toolkit array
        // use the top most toolkit max compute
        int minorVersion = embeddedDevice ? tkitMaxCompute->embedded_minor
                                          : tkitMaxCompute->minor;
        if (computeCapability.second > minorVersion) {
            computeCapability =
                make_pair(Toolkit2MaxCompute[0].major, minorVersion);
            spdlog::get("platform")
                ->warn(
                    "CUDA runtime version({}) not recognized. Targeting "
                    "compute {}.{} for this device which is the latest compute "
                    "capability supported by ArrayFire's CUDA runtime({}.{}). "
                    "Please create an issue or a pull request on the ArrayFire "
                    "repository to update the Toolkit2MaxCompute array with "
                    "this version of the CUDA Runtime.",
                    fromCudaVersion(rtCudaVer), originalCompute.first,
                    originalCompute.second, computeCapability.first,
                    computeCapability.second, computeCapability.first,
                    computeCapability.second);
        }
    } else if (computeCapability.first < 3) {
        // all compute versions prior to Kepler, we don't support
        // don't change the computeCapability.
        spdlog::get("platform")
            ->warn(
                "The compute capability of the current device({}.{}) "
                "lower than the minimum compute version ArrayFire "
                "supports.",
                originalCompute.first, originalCompute.second);
    }
}

pair<int, int> getComputeCapability(const int device) {
    return DeviceManager::getInstance().devJitComputes[device];
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
    static auto *my_instance = new DeviceManager();
    return *my_instance;
}

void DeviceManager::setMemoryManager(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a memory manager and the default memory
    // manager still hasn't been initialized, so initialize it anyways so we
    // don't inadvertently reset to it when we first call memoryManager()
    memoryManager();
    // Calls shutdown() on the existing memory manager.
    if (memManager) { memManager->shutdownAllocator(); }
    memManager = std::move(newMgr);
    // Set the backend memory manager for this new manager to register native
    // functions correctly.
    std::unique_ptr<cuda::Allocator> deviceMemoryManager(new Allocator());
    memManager->setAllocator(std::move(deviceMemoryManager));
    memManager->initialize();
}

void DeviceManager::resetMemoryManager() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(
        new common::DefaultMemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                         AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG));
    setMemoryManager(std::move(mgr));
}

void DeviceManager::setMemoryManagerPinned(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a pinned memory manager and the default
    // memory manager still hasn't been initialized, so initialize it anyways so
    // we don't inadvertently reset to it when we first call
    // pinnedMemoryManager()
    pinnedMemoryManager();
    // Calls shutdown() on the existing memory manager.
    if (pinnedMemManager) { pinnedMemManager->shutdownAllocator(); }
    // Set the backend memory manager for this new manager to register native
    // functions correctly.
    pinnedMemManager = std::move(newMgr);
    std::unique_ptr<cuda::AllocatorPinned> deviceMemoryManager(
        new AllocatorPinned());
    pinnedMemManager->setAllocator(std::move(deviceMemoryManager));
    pinnedMemManager->initialize();
}

void DeviceManager::resetMemoryManagerPinned() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(
        new common::DefaultMemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                         AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG));
    setMemoryManagerPinned(std::move(mgr));
}

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
void debugRuntimeCheck(spdlog::logger *logger, int runtime_version,
                       int driver_version) {
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

    auto getLogger = [&logger]() -> spdlog::logger * { return logger; };

    // If the runtime version is not part of the CudaToDriverVersion array,
    // display a message in the trace. Do not throw an error unless this is
    // a debug build
    if (runtime_it == end(CudaToDriverVersion)) {
        constexpr size_t buf_size = 256;
        char buf[buf_size];
        const char *err_msg =
            "CUDA runtime version({}) not recognized. Please create an issue "
            "or a pull request on the ArrayFire repository to update the "
            "CudaToDriverVersion variable with this version of the CUDA "
            "runtime.\n";
        fmt::format_to_n(buf, buf_size, err_msg,
                         fromCudaVersion(runtime_version));
        AF_TRACE("{}", buf);
#ifndef NDEBUG
        AF_ERROR(buf, AF_ERR_RUNTIME);
#endif
    }

    if (driver_it == end(CudaToDriverVersion)) {
        AF_TRACE(
            "CUDA driver version({}) not part of the CudaToDriverVersion "
            "array. Please create an issue or a pull request on the ArrayFire "
            "repository to update the CudaToDriverVersion variable with this "
            "version of the CUDA runtime.\n",
            fromCudaVersion(driver_version));
    }
}

// Check if the device driver version is recent enough to run the cuda libs
// linked with afcuda:
void DeviceManager::checkCudaVsDriverVersion() {
    const std::string driverVersionString = getDriverVersion();

    int driver  = 0;
    int runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

    AF_TRACE("CUDA Driver supports up to CUDA {} ArrayFire CUDA Runtime {}",
             fromCudaVersion(driver), fromCudaVersion(runtime));

    debugRuntimeCheck(getLogger(), runtime, driver);

    if (runtime > driver) {
        string msg =
            "ArrayFire was built with CUDA {} which requires GPU driver "
            "version {} or later. Please download and install the latest "
            "drivers from https://www.nvidia.com/drivers for your GPU. "
            "Alternatively, you could rebuild ArrayFire with CUDA Toolkit "
            "version {} to use the current drivers.";

        auto runtime_it =
            find_if(begin(CudaToDriverVersion), end(CudaToDriverVersion),
                    [runtime](ToolkitDriverVersions ver) {
                        return runtime == ver.version;
                    });

        constexpr size_t buf_size = 1024;
        // If the runtime version is not part of the CudaToDriverVersion
        // array, display a message in the trace. Do not throw an error
        // unless this is a debug build
        if (runtime_it == end(CudaToDriverVersion)) {
            char buf[buf_size];
            char err_msg[] =
                "CUDA runtime version(%s) not recognized. Please create an "
                "issue or a pull request on the ArrayFire repository to "
                "update the CudaToDriverVersion variable with this "
                "version of the CUDA Toolkit.";
            snprintf(buf, buf_size, err_msg,
                     fmt::format("{}", fromCudaVersion(runtime)).c_str());
            AF_TRACE("{}", buf);
            return;
        }

        float minimumDriverVersion =
#ifdef OS_WIN
            runtime_it->windows_min_version;
#else
            runtime_it->unix_min_version;
#endif

        char buf[buf_size];
        fmt::format_to_n(buf, buf_size, msg, fromCudaVersion(runtime),
                         minimumDriverVersion, fromCudaVersion(driver));

        AF_ERROR(buf, AF_ERR_DRIVER);
    }
}

/// This function initializes and deletes a nvrtcProgram object. There seems to
/// be a bug in nvrtc which fails if this is first done on a child thread. We
/// are assuming that the initilization is done in the main thread.
void initNvrtc() {
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, " ", "dummy", 0, nullptr, nullptr);
    nvrtcDestroyProgram(&prog);
}

DeviceManager::DeviceManager()
    : logger(common::loggerFactory("platform"))
    , cuDevices(0)
    , nDevices(0)
    , fgMngr(new arrayfire::common::ForgeManager()) {
    try {
        checkCudaVsDriverVersion();

        CUDA_CHECK(cudaGetDeviceCount(&nDevices));
        AF_TRACE("Found {} CUDA devices", nDevices);
        if (nDevices == 0) {
            AF_ERROR("No CUDA capable devices found", AF_ERR_DRIVER);
            return;
        }
        cuDevices.reserve(nDevices);

        int cudaRtVer = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&cudaRtVer));
        int cudaMajorVer = cudaRtVer / 1000;

        for (int i = 0; i < nDevices; i++) {
            cudaDevice_t dev{};
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
                    "Found device: {} (sm_{}{}) ({:0.3} GB | ~{} GFLOPs | {} "
                    "SMs)",
                    dev.prop.name, dev.prop.major, dev.prop.minor,
                    dev.prop.totalGlobalMem / 1024. / 1024. / 1024.,
                    dev.flops / 1024. / 1024. * 2,
                    dev.prop.multiProcessorCount);
                cuDevices.push_back(dev);
            }
        }
    } catch (const AfError &err) {
        // If one of the CUDA functions threw an exception. catch it and wrap it
        // into a more informative ArrayFire exception.
        if (err.getError() == AF_ERR_INTERNAL) {
            AF_ERROR(
                "Error initializing CUDA runtime. Check your CUDA device is "
                "visible to the OS and you have installed the correct driver. "
                "Try running the nvidia-smi utility to debug any driver "
                "issues.",
                AF_ERR_RUNTIME);
        } else {
            throw;
        }
    }
    nDevices = cuDevices.size();

    sortDevices();

    // Set all default peer access to false
    for (auto &dev_map : device_peer_access_map)
        for (auto &dev_access : dev_map) { dev_access = false; }

    // Enable peer 2 peer access to device memory if available
    for (int i = 0; i < nDevices; i++) {
        for (int j = 0; j < nDevices; j++) {
            if (i != j) {
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, i, j));
                if (can_access_peer) {
                    CUDA_CHECK(cudaSetDevice(i));
                    AF_TRACE("Peer access enabled for {}({}) and {}({})", i,
                             cuDevices[i].prop.name, j, cuDevices[j].prop.name);
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
                    device_peer_access_map[i][j] = true;
                }
            } else {
                device_peer_access_map[i][j] = true;
            }
        }
    }

    // Initialize all streams to 0.
    // Streams will be created in setActiveDevice()
    for (int i = 0; i < MAX_DEVICES; i++) {
        streams[i] = static_cast<cudaStream_t>(0);
        if (i < nDevices) {
            auto prop =
                make_pair(cuDevices[i].prop.major, cuDevices[i].prop.minor);
            checkAndSetDevMaxCompute(prop);
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
    initNvrtc();
    AF_TRACE("Default device: {}({})", getActiveDeviceId(),
             cuDevices[getActiveDeviceId()].prop.name);
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

    if (device >= numDevices) { return -1; }

    int old = getActiveDeviceId();

    if (nId == -1) { nId = getDeviceNativeId(device); }

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
        if (device >= numDevices) { break; }

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
}  // namespace arrayfire
