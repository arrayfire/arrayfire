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

#include <common/Logger.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <driver.h>
#include <err_cuda.hpp>
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
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using std::to_string;

namespace cuda {

void findJitDevCompute(pair<int, int>& prop) {
    struct cuNVRTCcompute {
        /// The CUDA Toolkit version returned by cudaRuntimeGetVersion
        int cuda_version;
        /// Maximum major compute flag supported by cuda_version
        int major;
        /// Maximum minor compute flag supported by cuda_version
        int minor;
    };
    static const cuNVRTCcompute Toolkit2Compute[] = {
        {10010, 7, 5},
        {10000, 7, 2},
        {9020,  7, 2},
        {9010,  7, 2},
        {9000,  7, 2},
        {8000,  5, 3},
        {7050,  5, 3},
        {7000,  5, 3}
    };
    int runtime_cuda_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_cuda_ver));
    auto tkit_max_compute =
        find_if(begin(Toolkit2Compute), end(Toolkit2Compute),
                [runtime_cuda_ver](cuNVRTCcompute v) {
                    return runtime_cuda_ver == v.cuda_version;
                });
    if (tkit_max_compute == end(Toolkit2Compute)) {
        // Unidentified Runtime CUDA toolkit version
        // return default compute flag 30
        prop = make_pair(3, 0);
    } else if (prop.first  > tkit_max_compute->major &&
               prop.second > tkit_max_compute->minor) {
        prop = make_pair(tkit_max_compute->major, tkit_max_compute->minor);
    }
}

pair<int, int> getComputeFlag() {
    return DeviceManager::getInstance().devJitComputes[getActiveDeviceId()];
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

int &tlocalActiveDeviceId() {
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
    thread_local PlanCache cufftManagers[DeviceManager::MAX_DEVICES];

    return cufftManagers[getActiveDeviceId()];
}

BlasHandle blasHandle() {
    thread_local std::unique_ptr<cublasHandle>
        cublasHandles[DeviceManager::MAX_DEVICES];
    thread_local std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = cuda::getActiveDeviceId();

    std::call_once(initFlags[id],
                   [&] { cublasHandles[id].reset(new cublasHandle()); });

    CUBLAS_CHECK(
        cublasSetStream(cublasHandles[id].get()->get(), cuda::getStream(id)));

    return cublasHandles[id].get()->get();
}

SolveHandle solverDnHandle() {
    thread_local std::unique_ptr<cusolverDnHandle>
        cusolverHandles[DeviceManager::MAX_DEVICES];
    thread_local std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = cuda::getActiveDeviceId();

    std::call_once(initFlags[id],
                   [&] { cusolverHandles[id].reset(new cusolverDnHandle()); });

    // FIXME
    // This is not an ideal case. It's just a hack.
    // The correct way to do is to use
    // CUSOLVER_CHECK(cusolverDnSetStream(cuda::getStream(cuda::getActiveDeviceId())))
    // in the class constructor.
    // However, this is causing a lot of the cusolver functions to fail.
    // The only way to fix them is to use cudaDeviceSynchronize() and
    //     cudaStreamSynchronize()
    // all over the place, but even then some calls like getrs in solve_lu
    // continue to fail on any stream other than 0.
    //
    // cuSolver Streams patch:
    // https://gist.github.com/shehzan10/414c3d04a40e7c4a03ed3c2e1b9072e7
    CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(id)));

    return cusolverHandles[id].get()->get();
}

SparseHandle sparseHandle() {
    thread_local std::unique_ptr<cusparseHandle>
        cusparseHandles[DeviceManager::MAX_DEVICES];
    thread_local std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = cuda::getActiveDeviceId();

    std::call_once(initFlags[id],
                   [&] { cusparseHandles[id].reset(new cusparseHandle()); });

    CUSPARSE_CHECK(cusparseSetStream(cusparseHandles[id].get()->get(),
                                     cuda::getStream(id)));

    return cusparseHandles[id].get()->get();
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
        fprintf(stderr, err_msg,
                int_version_to_string(driver_version).c_str());
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
    , fgMngr(new graphics::ForgeManager())
    {
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
    for (int i = 0; i < (int)MAX_DEVICES; i++) {
        streams[i] = (cudaStream_t)0;
        if (i < nDevices) {
            auto prop = make_pair(cuDevices[i].prop.major,
                    cuDevices[i].prop.minor);
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
