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
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/defines.hpp>
#include <common/graphics_common.hpp>
#include <common/host_memory.hpp>
#include <common/unique_handle.hpp>
#include <common/util.hpp>
#include <cublas.hpp>
#include <cudnn.hpp>
#include <cudnnModule.hpp>
#include <cufft.hpp>
#include <cusolverDn.hpp>
#include <cusparse.hpp>
#include <device_manager.hpp>
#include <driver.h>
#include <err_cuda.hpp>
#include <memory.hpp>
#include <spdlog/spdlog.h>
#include <version.hpp>
#include <af/cuda.h>
#include <af/device.h>
#include <af/version.h>

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

using common::unique_handle;
using common::memory::MemoryManagerBase;

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

unique_handle<cublasHandle_t> *cublasManager(const int deviceId) {
    thread_local unique_handle<cublasHandle_t>
        handles[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];

    call_once(initFlags[deviceId], [&] {
        handles[deviceId].create();
        // TODO(pradeep) When multiple streams per device
        // is added to CUDA backend, move the cublasSetStream
        // call outside of call_once scope.
        CUBLAS_CHECK(
            cublasSetStream(handles[deviceId], cuda::getStream(deviceId)));
    });

    return &handles[deviceId];
}

unique_handle<cudnnHandle_t> *nnManager(const int deviceId) {
    thread_local unique_handle<cudnnHandle_t>
        cudnnHandles[DeviceManager::MAX_DEVICES];
    thread_local std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    auto *handle        = &cudnnHandles[deviceId];
    cudnnStatus_t error = CUDNN_STATUS_SUCCESS;
    std::call_once(initFlags[deviceId], [deviceId, handle, &error] {
        auto getLogger = [&] { return spdlog::get("platform"); };
        AF_TRACE("Initializing cuDNN");
        error = static_cast<cudnnStatus_t>(handle->create());

        // Not throwing an AF_ERROR here because we are in a lambda that could
        // be executing on another thread;
        if (!(*handle)) getLogger()->error("Error initalizing cuDNN");
    });
    if (error) {
        string error_msg = fmt::format("Error initializing cuDNN({}): {}.",
                                       error, errorString(error));
        AF_ERROR(error_msg, AF_ERR_RUNTIME);
    }
    CUDNN_CHECK(getCudnnPlugin().cudnnSetStream(cudnnHandles[deviceId],
                                                cuda::getStream(deviceId)));

    return handle;
}

unique_ptr<PlanCache> &cufftManager(const int deviceId) {
    thread_local unique_ptr<PlanCache> caches[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];
    call_once(initFlags[deviceId],
              [&] { caches[deviceId].reset(new PlanCache()); });
    return caches[deviceId];
}

unique_handle<cusolverDnHandle_t> *cusolverManager(const int deviceId) {
    thread_local unique_handle<cusolverDnHandle_t>
        handles[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];
    call_once(initFlags[deviceId], [&] {
        handles[deviceId].create();
        // TODO(pradeep) When multiple streams per device
        // is added to CUDA backend, move the cublasSetStream
        // call outside of call_once scope.
        CUSOLVER_CHECK(
            cusolverDnSetStream(handles[deviceId], cuda::getStream(deviceId)));
    });
    // TODO(pradeep) prior to this change, stream was being synced in get solver
    // handle because of some cusolver bug. Re-enable that if this change
    // doesn't work and sovler tests fail.
    // https://gist.github.com/shehzan10/414c3d04a40e7c4a03ed3c2e1b9072e7
    // cuSolver Streams patch:
    // CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(deviceId)));

    return &handles[deviceId];
}

unique_handle<cusparseHandle_t> *cusparseManager(const int deviceId) {
    thread_local unique_handle<cusparseHandle_t>
        handles[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];
    call_once(initFlags[deviceId], [&] {
        handles[deviceId].create();
        // TODO(pradeep) When multiple streams per device
        // is added to CUDA backend, move the cublasSetStream
        // call outside of call_once scope.
        CUSPARSE_CHECK(
            cusparseSetStream(handles[deviceId], cuda::getStream(deviceId)));
    });
    return &handles[deviceId];
}

DeviceManager::~DeviceManager() {
    // Reset unique_ptrs for all cu[BLAS | Sparse | Solver]
    // handles of all devices
    for (int i = 0; i < nDevices; ++i) {
        setDevice(i);
        delete cusolverManager(i);
        delete cusparseManager(i);
        cufftManager(i).reset();
        delete cublasManager(i);
        delete nnManager(i);
    }
}

int getBackend() { return AF_BACKEND_CUDA; }

string getDeviceInfo(int device) noexcept {
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

string getDeviceInfo() noexcept {
    ostringstream info;
    info << "ArrayFire v" << AF_VERSION << " (CUDA, " << get_system()
         << ", build " << AF_REVISION << ")" << std::endl;
    info << getPlatformInfo();
    for (int i = 0; i < getDeviceCount(); ++i) { info << getDeviceInfo(i); }
    return info.str();
}

string getPlatformInfo() noexcept {
    string driverVersion    = getDriverVersion();
    std::string cudaRuntime = getCUDARuntimeVersion();
    string platform         = "Platform: CUDA Runtime " + cudaRuntime;
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

bool isHalfSupported(int device) {
    auto prop     = getDeviceProp(device);
    float compute = prop.major * 1000 + prop.minor * 10;
    return compute >= 5030;
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
    char driverVersion[1024] = {" "};
    int x = nvDriverVersion(driverVersion, sizeof(driverVersion));
    if (x != 1) {
// Windows, OSX, Tegra Need a new way to fetch driver
#if !defined(OS_WIN) && !defined(OS_MAC) && !defined(__arm__) && \
    !defined(__aarch64__)
        return "N/A";
#endif
        int driver = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        return to_string(driver);
    } else {
        return string(driverVersion);
    }
}

string getCUDARuntimeVersion() noexcept {
    int runtime = 0;
    if (cudaSuccess == cudaRuntimeGetVersion(&runtime)) {
        return int_version_to_string(runtime);
    } else {
        return int_version_to_string(CUDA_VERSION);
    }
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

int getDeviceCount() {
    int count = 0;
    if (cudaGetDeviceCount(&count)) {
        return 0;
    } else {
        return count;
    }
}

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

MemoryManagerBase &memoryManager() {
    static std::once_flag flag;

    DeviceManager &inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() {
        // By default, create an instance of the default memory manager
        inst.memManager.reset(new common::DefaultMemoryManager(
            getDeviceCount(), common::MAX_BUFFERS,
            AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG));
        // Set the memory manager's device memory manager
        std::unique_ptr<cuda::Allocator> deviceMemoryManager(
            new cuda::Allocator());
        inst.memManager->setAllocator(std::move(deviceMemoryManager));
        inst.memManager->initialize();
    });

    return *(inst.memManager.get());
}

MemoryManagerBase &pinnedMemoryManager() {
    static std::once_flag flag;

    DeviceManager &inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() {
        // By default, create an instance of the default memory manager
        inst.pinnedMemManager.reset(new common::DefaultMemoryManager(
            getDeviceCount(), common::MAX_BUFFERS,
            AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG));
        // Set the memory manager's device memory manager
        std::unique_ptr<cuda::AllocatorPinned> deviceMemoryManager(
            new cuda::AllocatorPinned());
        inst.pinnedMemManager->setAllocator(std::move(deviceMemoryManager));
        inst.pinnedMemManager->initialize();
    });

    return *(inst.pinnedMemManager.get());
}

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManager(std::move(mgr));
}

void resetMemoryManager() {
    return DeviceManager::getInstance().resetMemoryManager();
}

void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManagerPinned(std::move(mgr));
}

void resetMemoryManagerPinned() {
    return DeviceManager::getInstance().resetMemoryManagerPinned();
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

BlasHandle blasHandle() { return *cublasManager(cuda::getActiveDeviceId()); }

cudnnHandle_t nnHandle() {
    // Keep the getCudnnPlugin call here because module loading can throw an
    // exception the first time its called. We want to avoid that because the
    // unique handle object is marked noexcept and could terminate. if the
    // module is not loaded correctly
    static cudnnModule keep_me_to_avoid_exceptions_exceptions =
        getCudnnPlugin();
    static unique_handle<cudnnHandle_t> *handle =
        nnManager(cuda::getActiveDeviceId());
    if (*handle)
        return *handle;
    else {
        AF_ERROR("Error Initializing cuDNN\n", AF_ERR_RUNTIME);
    }
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

af_err afcu_cublasSetMathMode(cublasMath_t mode) {
    try {
        CUBLAS_CHECK(cublasSetMathMode(cuda::blasHandle(), mode));
    }
    CATCHALL;
    return AF_SUCCESS;
}

namespace af {
template<>
__half *array::device<__half>() const {
    void *ptr = NULL;
    af_get_device_ptr(&ptr, get());
    return (__half *)ptr;
}
}  // namespace af
