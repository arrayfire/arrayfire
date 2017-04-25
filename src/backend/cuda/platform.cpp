/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/version.h>
#include <af/cuda.h>
#include <platform.hpp>
#include <defines.hpp>
#include <util.hpp>
#include <version.hpp>
#include <driver.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <err_cuda.hpp>
#include <util.hpp>
#include <host_memory.hpp>

using namespace std;

namespace cuda
{
///////////////////////////////////////////////////////////////////////////
// HELPERS
///////////////////////////////////////////////////////////////////////////
// pulled from CUTIL from CUDA SDK
static inline int compute2cores(int major, int minor)
{
    struct {
        int compute; // 0xMm (hex), M = major version, m = minor version
        int cores;
    } gpus[] = {
        { 0x10,  8 },
        { 0x11,  8 },
        { 0x12,  8 },
        { 0x13,  8 },
        { 0x20, 32 },
        { 0x21, 48 },
        { 0x30, 192 },
        { 0x32, 192 },
        { 0x35, 192 },
        { 0x37, 192 },
        { 0x50, 128 },
        { 0x52, 128 },
        { 0x53, 128 },
        { 0x60, 128 },
        { 0x61, 64  },
        { 0x62, 128 },
        {   -1, -1  },
    };

    for (int i = 0; gpus[i].compute != -1; ++i) {
        if (gpus[i].compute == (major << 4) + minor)
            return gpus[i].cores;
    }
    return 0;
}

// Return true if greater, false if lesser.
// if equal, it continues to next comparison
#define COMPARE(a,b,f) do {                     \
        if ((a)->f > (b)->f) return true;       \
        if ((a)->f < (b)->f) return false;      \
        break;                                  \
    } while (0)


static inline bool card_compare_compute(const cudaDevice_t &l, const cudaDevice_t &r)
{
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, prop.major);
    COMPARE(lc, rc, prop.minor);
    COMPARE(lc, rc, flops);
    COMPARE(lc, rc, prop.totalGlobalMem);
    COMPARE(lc, rc, nativeId);
    return false;
}

static inline bool card_compare_flops(const cudaDevice_t &l, const cudaDevice_t &r)
{
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, flops);
    COMPARE(lc, rc, prop.totalGlobalMem);
    COMPARE(lc, rc, prop.major);
    COMPARE(lc, rc, prop.minor);
    COMPARE(lc, rc, nativeId);
    return false;
}

static inline bool card_compare_mem(const cudaDevice_t &l, const cudaDevice_t &r)
{
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, prop.totalGlobalMem);
    COMPARE(lc, rc, flops);
    COMPARE(lc, rc, prop.major);
    COMPARE(lc, rc, prop.minor);
    COMPARE(lc, rc, nativeId);
    return false;
}

static inline bool card_compare_num(const cudaDevice_t &l, const cudaDevice_t &r)
{
    const cudaDevice_t *lc = &l;
    const cudaDevice_t *rc = &r;

    COMPARE(lc, rc, nativeId);
    return false;
}

static const std::string get_system(void)
{
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

template <typename T>
static inline string toString(T val)
{
    stringstream s;
    s << val;
    return s.str();
}

///////////////////////////////////////////////////////////////////////////
// Wrapper Functions
///////////////////////////////////////////////////////////////////////////
int getBackend()
{
    return AF_BACKEND_CUDA;
}

string getDeviceInfo(int device)
{
    cudaDeviceProp dev = getDeviceProp(device);

    size_t mem_gpu_total = dev.totalGlobalMem;
    //double cc = double(dev.major) + double(dev.minor) / 10;

    bool show_braces = getActiveDeviceId() == device;

    string id = (show_braces ? string("[") : "-") + toString(device) +
                (show_braces ? string("]") : "-");
    string name(dev.name);
    string memory = toString((mem_gpu_total / (1024 * 1024))
                          + !!(mem_gpu_total % (1024 * 1024)))
                    + string(" MB");
    string compute = string("CUDA Compute ") + toString(dev.major) + string(".") + toString(dev.minor);

    string info = id + string(" ")  +
                name + string(", ") +
              memory + string(", ") +
             compute + string("\n");
    return info;
}

string getDeviceInfo()
{
    ostringstream info;
    info << "ArrayFire v" << AF_VERSION
         << " (CUDA, " << get_system() << ", build " << AF_REVISION << ")" << std::endl;
    info << getPlatformInfo();
    for (int i = 0; i < getDeviceCount(); ++i) {
        info << getDeviceInfo(i);
    }
    return info.str();
}

string getPlatformInfo()
{
    string driverVersion = getDriverVersion();
    std::string cudaRuntime = getCUDARuntimeVersion();
    string platform = "Platform: CUDA Toolkit " + cudaRuntime;
    if (!driverVersion.empty()) {
        platform.append(", Driver: ");
        platform.append(driverVersion);
    }
    platform.append("\n");
    return platform;
}

bool isDoubleSupported(int device)
{
    return true;
}

void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    if (getDeviceCount() <= 0) {
        printf("No CUDA-capable devices detected.\n");
        return;
    }

    cudaDeviceProp dev = getDeviceProp(getActiveDeviceId());

    // Name
    snprintf(d_name, 64, "%s", dev.name);

    //Platform
    std::string cudaRuntime = getCUDARuntimeVersion();
    snprintf(d_platform, 10, "CUDA");
    snprintf(d_toolkit, 64, "v%s", cudaRuntime.c_str());

    // Compute Version
    snprintf(d_compute, 10, "%d.%d", dev.major, dev.minor);

    // Sanitize input
    for (int i = 0; i < 63; i++) {
        if (d_name[i] == ' ') {
            if (d_name[i + 1] == 0 || d_name[i + 1] == ' ') d_name[i] = 0;
            else d_name[i] = '_';
        }
    }
}

string getDriverVersion()
{
    char driverVersion[1024] = {" ",};
    int x = nvDriverVersion(driverVersion, sizeof(driverVersion));
    if (x != 1) {
        // Windows, OSX, Tegra Need a new way to fetch driver
        #if !defined(OS_WIN) && !defined(OS_MAC) && !defined(__arm__) && !defined(__aarch64__)
        throw runtime_error("Invalid driver");
        #endif
        int driver = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        return string("CUDA Driver Version: ") + toString(driver);
    } else {
        return string(driverVersion);
    }
}

string getCUDARuntimeVersion()
{
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    if(runtime / 100.f > 0)
        return toString((runtime / 1000) + (runtime % 1000)/ 100.);
    else
        return toString(runtime / 1000) + string(".0");

}

unsigned getMaxJitSize()
{
    const int MAX_JIT_LEN = 100;

    static int length = 0;
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

int& tlocalActiveDeviceId()
{
    thread_local static int activeDeviceId = 0;

    return activeDeviceId;
}

int getDeviceCount()
{
    return DeviceManager::getInstance().nDevices;
}

int getActiveDeviceId()
{
    return tlocalActiveDeviceId();
}

int getDeviceNativeId(int device)
{
    if(device < (int)DeviceManager::getInstance().cuDevices.size())
        return DeviceManager::getInstance().cuDevices[device].nativeId;
    return -1;
}

int getDeviceIdFromNativeId(int nativeId)
{
    DeviceManager& mngr = DeviceManager::getInstance();

    int devId = 0;
    for(devId = 0; devId < mngr.nDevices; ++devId) {
        if (nativeId == mngr.cuDevices[devId].nativeId)
            break;
    }
    return devId;
}

cudaStream_t DeviceManager::nextAvailableStream(int device)
{
    common::lock_guard_t lock(poolCounterMutexes[device]);

    unsigned oldPoolId = nextPoolCounter[device];

    nextPoolCounter[device] = (oldPoolId+1) % streamPoolCluster[device].size();

    return streamPoolCluster[device][oldPoolId];
}

cudaStream_t getStream(int device)
{
    static thread_local cudaStream_t myDefaultStream =
        DeviceManager::getInstance().nextAvailableStream(device);

    return myDefaultStream;
}

cudaStream_t getActiveStream()
{
    return getStream(getActiveDeviceId());
}

size_t getDeviceMemorySize(int device)
{
    return getDeviceProp(device).totalGlobalMem;
}

size_t getHostMemorySize()
{
    return common::getHostMemorySize();
}

int setDevice(int device)
{
    return DeviceManager::getInstance().setActiveDevice(device);
}

cudaDeviceProp getDeviceProp(int device)
{
    if(device < (int)DeviceManager::getInstance().cuDevices.size())
        return DeviceManager::getInstance().cuDevices[device].prop;
    return DeviceManager::getInstance().cuDevices[0].prop;
}

///////////////////////////////////////////////////////////////////////////
// DeviceManager Class Functions
///////////////////////////////////////////////////////////////////////////
#if defined(WITH_GRAPHICS)
bool DeviceManager::checkGraphicsInteropCapability()
{
    static bool run_once = true;
    static bool capable  = true;

    if(run_once) {
        unsigned int pCudaEnabledDeviceCount = 0;
        int pCudaGraphicsEnabledDeviceIds = 0;
        cudaGetLastError(); // Reset Errors
        cudaError_t err = cudaGLGetDevices(&pCudaEnabledDeviceCount, &pCudaGraphicsEnabledDeviceIds, getDeviceCount(), cudaGLDeviceListAll);
        if(err == 63) { // OS Support Failure - Happens when devices are only Tesla
            capable = false;
            printf("Warning: No CUDA Device capable of CUDA-OpenGL. CUDA-OpenGL Interop will use CPU fallback.\n");
            printf("Corresponding CUDA Error (%d): %s.\n", err, cudaGetErrorString(err));
            printf("This may happen if all CUDA Devices are in TCC Mode and/or not connected to a display.\n");
        }
        cudaGetLastError(); // Reset Errors
        run_once = false;
    }

    return capable;
}
#endif

DeviceManager& DeviceManager::getInstance()
{
    static DeviceManager my_instance;
    return my_instance;
}

MemoryManager& memoryManager()
{
    static std::once_flag flag;

    DeviceManager& inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() { inst.memManager.reset(new MemoryManager()); });

    return *(inst.memManager.get());
}

MemoryManagerPinned& pinnedMemoryManager()
{
    static std::once_flag flag;

    DeviceManager& inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() { inst.pinnedMemManager.reset(new MemoryManagerPinned()); });

    return *(inst.pinnedMemManager.get());
}

#if defined(WITH_GRAPHICS)
GraphicsResourceManager& interopManager()
{
    static std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = getActiveDeviceId();

    DeviceManager& inst = DeviceManager::getInstance();

    std::call_once(initFlags[id], [&]{ inst.gfxManagers[id].reset(new GraphicsResourceManager()); });

    return *(inst.gfxManagers[id].get());
}
#endif

PlanCache& fftManager()
{
    thread_local static PlanCache cufftManagers[DeviceManager::MAX_DEVICES];

    return cufftManagers[getActiveDeviceId()];
}

BlasHandle blasHandle()
{
    thread_local static std::unique_ptr<cublasHandle> cublasHandles[DeviceManager::MAX_DEVICES];
    thread_local static std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = cuda::getActiveDeviceId();

    std::call_once(initFlags[id], [&]{ cublasHandles[id].reset(new cublasHandle()); });

    CUBLAS_CHECK(cublasSetStream(cublasHandles[id].get()->get(), cuda::getStream(id)));

    return cublasHandles[id].get()->get();
}

SolveHandle solverDnHandle()
{
    thread_local static std::unique_ptr<cusolverDnHandle> cusolverHandles[DeviceManager::MAX_DEVICES];
    thread_local static std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = cuda::getActiveDeviceId();

    std::call_once(initFlags[id], [&]{ cusolverHandles[id].reset(new cusolverDnHandle()); });

    //FIXME
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

SparseHandle sparseHandle()
{
    thread_local static std::unique_ptr<cusparseHandle> cusparseHandles[DeviceManager::MAX_DEVICES];
    thread_local static std::once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = cuda::getActiveDeviceId();

    std::call_once(initFlags[id], [&]{ cusparseHandles[id].reset(new cusparseHandle()); });

    CUSPARSE_CHECK(cusparseSetStream(cusparseHandles[id].get()->get(), cuda::getStream(id)));

    return cusparseHandles[id].get()->get();
}

DeviceManager::DeviceManager()
    : cuDevices(0), nDevices(0)
{
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices == 0)
        throw runtime_error("No CUDA-Capable devices found");

    cuDevices.reserve(nDevices);

    for(int i = 0; i < nDevices; i++) {
        cudaDevice_t dev;
        cudaGetDeviceProperties(&dev.prop, i);
        dev.flops = dev.prop.multiProcessorCount * compute2cores(dev.prop.major, dev.prop.minor) * dev.prop.clockRate;
        dev.nativeId = i;
        cuDevices.push_back(dev);
    }

    sortDevices();

    // Initialize stream pools for all devices.
    for (unsigned c=0; c < streamPoolCluster.size(); ++c) {
        streamPoolCluster[c].resize(MAX_SIZE_STREAM_POOL, static_cast<cudaStream_t>(0));
        for (unsigned p=0; p<MAX_SIZE_STREAM_POOL; ++p) {
            cudaStream_t temp = 0;
            CUDA_CHECK(cudaStreamCreate(&temp));
            streamPoolCluster[c][p] = temp;
        }
    }
    nextPoolCounter.fill(0);

    std::string deviceENV = getEnvVar("AF_CUDA_DEFAULT_DEVICE");
    if(deviceENV.empty()) {
        setActiveDevice(0, cuDevices[0].nativeId);
    } else {
        stringstream s(deviceENV);
        int def_device = -1;
        s >> def_device;
        if(def_device < 0 || def_device >= nDevices) {
            printf("WARNING: AF_CUDA_DEFAULT_DEVICE is out of range\n");
            printf("Setting default device as 0\n");
            setActiveDevice(0, cuDevices[0].nativeId);
        } else {
            setActiveDevice(def_device, cuDevices[def_device].nativeId);
        }
    }
}

void DeviceManager::sortDevices(sort_mode mode)
{
    switch(mode) {
        case memory :
            std::stable_sort(cuDevices.begin(), cuDevices.end(), card_compare_mem);
            break;
        case flops :
            std::stable_sort(cuDevices.begin(), cuDevices.end(), card_compare_flops);
            break;
        case compute :
            std::stable_sort(cuDevices.begin(), cuDevices.end(), card_compare_compute);
            break;
        case none : default :
            std::stable_sort(cuDevices.begin(), cuDevices.end(), card_compare_num);
            break;
    }
}

int DeviceManager::setActiveDevice(int device, int nId)
{
    thread_local static bool retryFlag = true;

    int numDevices = cuDevices.size();

    if (device > numDevices)
        return -1;

    int old = getActiveDeviceId();

    if(nId == -1)
        nId = getDeviceNativeId(device);

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

    while(true) {
        // Check for errors other than DevicesUnavailable
        // If success, return. Else throw error
        // If DevicesUnavailable, try other devices (while loop below)
        if (err != cudaErrorDeviceAlreadyInUse) {
            CUDA_CHECK(err);
            tlocalActiveDeviceId() = device;
            return old;
        }
        cudaGetLastError(); // Reset error stack
#ifndef NDEBUG
        printf("Warning: Device %d is unavailable. Incrementing to next device \n", device);
#endif
        // Comes here is the device is in exclusive mode or
        // otherwise fails streamCreate with this error.
        // All other errors will error out
        device++;
        if (device >= numDevices) break;

        // Can't call getNativeId here as it will cause an infinite loop with the constructor
        nId = cuDevices[device].nativeId;

        err = cudaSetDevice(nId);
    }

    // If all devices fail with DeviceAlreadyInUse, then throw this error
    CUDA_CHECK(err);

    return old;
}

void sync(int device)
{
    int currDevice = getActiveDeviceId();
    setDevice(device);
    CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));
    setDevice(currDevice);
}

bool synchronize_calls()
{
    static bool sync = getEnvVar("AF_SYNCHRONOUS_CALLS") == "1";
    return sync;
}

bool& evalFlag()
{
    static bool flag = true;
    return flag;
}
}

af_err afcu_get_stream(cudaStream_t* stream, int id)
{
    try{
        *stream = cuda::getStream(id);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcu_get_native_id(int* nativeid, int id)
{
    try {
        *nativeid = cuda::getDeviceNativeId(id);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcu_set_native_id(int nativeid)
{
    try {
        cuda::setDevice(cuda::getDeviceIdFromNativeId(nativeid));
    } CATCHALL;
    return AF_SUCCESS;
}
