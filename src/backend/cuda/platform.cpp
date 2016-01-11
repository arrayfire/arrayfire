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

string getInfo()
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
    snprintf(d_name, 32, "%s", dev.name);

    //Platform
    std::string cudaRuntime = getCUDARuntimeVersion();
    snprintf(d_platform, 10, "CUDA");
    snprintf(d_toolkit, 64, "v%s", cudaRuntime.c_str());

    // Compute Version
    snprintf(d_compute, 10, "%d.%d", dev.major, dev.minor);

    // Sanitize input
    for (int i = 0; i < 31; i++) {
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
        #if !defined(OS_WIN) && !defined(OS_MAC) && !defined(__arm__)
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

int getDeviceCount()
{
    return DeviceManager::getInstance().nDevices;
}

int getActiveDeviceId()
{
    return DeviceManager::getInstance().activeDev;
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

cudaStream_t getStream(int device)
{
    cudaStream_t str = DeviceManager::getInstance().streams[device];
    // if the stream has not yet been initialized, ie. the device has not been
    // set to active at least once (cuz that's where the stream is created)
    // then set the device, get the stream, reset the device to current
    if(!str) {
        int active_dev = DeviceManager::getInstance().activeDev;
        setDevice(device);
        str = DeviceManager::getInstance().streams[device];
        setDevice(active_dev);
    }
    return str;
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
DeviceManager& DeviceManager::getInstance()
{
    static DeviceManager my_instance;
    return my_instance;
}

DeviceManager::DeviceManager()
    : cuDevices(0), activeDev(0), nDevices(0)
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

    // Initialize all streams to 0.
    // Streams will be created in setActiveDevice()
    for(int i = 0; i < (int)MAX_DEVICES; i++)
        streams[i] = (cudaStream_t)0;

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
    static bool first = true;

    int numDevices = cuDevices.size();

    if(device > numDevices) return -1;

    int old = activeDev;
    if(nId == -1) nId = getDeviceNativeId(device);
    CUDA_CHECK(cudaSetDevice(nId));
    cudaError_t err = cudaStreamCreate(&streams[device]);
    activeDev = device;

    if (err == cudaSuccess) return old;

    // Comes when user sets device
    // If success, return. Else throw error
    if (!first) {
        CUDA_CHECK(err);
        return old;
    }

    // Comes only when first is true. Set it to false
    first = false;

    while(device < numDevices) {
        // Check for errors other than DevicesUnavailable
        // If success, return. Else throw error
        // If DevicesUnavailable, try other devices (while loop below)
        if (err != cudaErrorDevicesUnavailable) {
            CUDA_CHECK(err);
            activeDev = device;
            return old;
        }
        cudaGetLastError(); // Reset error stack
        printf("Warning: Device %d is unavailable. Incrementing to next device \n", device);

        // Comes here is the device is in exclusive mode or
        // otherwise fails streamCreate with this error.
        // All other errors will error out
        device++;

        // Can't call getNativeId here as it will cause an infinite loop with the constructor
        nId = cuDevices[device].nativeId;

        CUDA_CHECK(cudaSetDevice(nId));
        err = cudaStreamCreate(&streams[device]);
    }

    // If all devices fail with DevicesUnavailable, then throw this error
    CUDA_CHECK(err);

    return old;
}

void sync(int device)
{
    int currDevice = getActiveDeviceId();
    setDevice(device);
    CUDA_CHECK(cudaStreamSynchronize(getStream(getActiveDeviceId())));
    setDevice(currDevice);
}

bool synchronize_calls() {
    static bool sync = getEnvVar("AF_SYNCHRONOUS_CALLS") == "1";
    return sync;
}

}

af_err afcu_get_stream(cudaStream_t* stream, int id)
{
    *stream = cuda::getStream(id);
    return AF_SUCCESS;
}

af_err afcu_get_native_id(int* nativeid, int id)
{
    *nativeid = cuda::getDeviceNativeId(id);
    return AF_SUCCESS;
}

af_err afcu_set_native_id(int nativeid)
{
    cuda::setDevice(cuda::getDeviceIdFromNativeId(nativeid));
    return AF_SUCCESS;
}
