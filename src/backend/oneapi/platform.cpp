/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <platform.hpp>

#include <GraphicsResourceManager.hpp>
#include <blas.hpp>
#include <build_version.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/graphics_common.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <err_oneapi.hpp>
#include <errorcodes.hpp>
#include <memory.hpp>
#include <onefft.hpp>
#include <af/oneapi.h>
#include <af/version.h>

#ifdef OS_MAC
#include <OpenGL/CGLCurrent.h>
#endif

#include <sycl/sycl.hpp>

#include <cctype>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using sycl::aspect;
using sycl::context;
using sycl::device;
using sycl::platform;
using sycl::queue;

using std::begin;
using std::call_once;
using std::end;
using std::endl;
using std::find_if;
using std::get;
using std::make_pair;
using std::make_unique;
using std::map;
using std::move;
using std::once_flag;
using std::ostringstream;
using std::pair;
using std::string;
using std::to_string;
using std::unique_ptr;
using std::vector;

using arrayfire::common::getEnvVar;
using arrayfire::common::ltrim;
using arrayfire::common::MemoryManagerBase;
using arrayfire::oneapi::Allocator;
using arrayfire::oneapi::AllocatorPinned;

namespace arrayfire {
namespace oneapi {

static string get_system() {
    string arch = (sizeof(void*) == 4) ? "32-bit " : "64-bit ";

    return arch +
#if defined(OS_LNX)
           "Linux";
#elif defined(OS_WIN)
           "Windows";
#elif defined(OS_MAC)
           "Mac OSX";
#endif
}

int getBackend() { return AF_BACKEND_ONEAPI; }

bool verify_present(const string& pname, const string ref) {
    auto iter =
        search(begin(pname), end(pname), begin(ref), end(ref),
               [](const string::value_type& l, const string::value_type& r) {
                   return tolower(l) == tolower(r);
               });

    return iter != end(pname);
}

// TODO: update to new platforms?
inline string platformMap(string& platStr) {
    using strmap_t                = map<string, string>;
    static const strmap_t platMap = {
        make_pair("NVIDIA CUDA", "NVIDIA"),
        make_pair("Intel(R) OpenCL", "INTEL"),
        make_pair("AMD Accelerated Parallel Processing", "AMD"),
        make_pair("Intel Gen OCL Driver", "BEIGNET"),
        make_pair("Intel(R) OpenCL HD Graphics", "INTEL"),
        make_pair("Apple", "APPLE"),
        make_pair("Portable Computing Language", "POCL"),
    };

    auto idx = platMap.find(platStr);

    if (idx == platMap.end()) {
        return platStr;
    } else {
        return idx->second;
    }
}

af_oneapi_platform getPlatformEnum(sycl::device dev) {
    string pname = getPlatformName(dev);
    if (verify_present(pname, "AMD"))
        return AF_ONEAPI_PLATFORM_AMD;
    else if (verify_present(pname, "NVIDIA"))
        return AF_ONEAPI_PLATFORM_NVIDIA;
    else if (verify_present(pname, "INTEL"))
        return AF_ONEAPI_PLATFORM_INTEL;
    else if (verify_present(pname, "APPLE"))
        return AF_ONEAPI_PLATFORM_APPLE;
    else if (verify_present(pname, "BEIGNET"))
        return AF_ONEAPI_PLATFORM_BEIGNET;
    else if (verify_present(pname, "POCL"))
        return AF_ONEAPI_PLATFORM_POCL;
    return AF_ONEAPI_PLATFORM_UNKNOWN;
}

string getDeviceInfo() noexcept {
    ostringstream info;
    info << "ArrayFire v" << AF_VERSION << " (oneAPI, " << get_system()
         << ", build " << AF_REVISION << ")\n";

    try {
        DeviceManager& devMngr = DeviceManager::getInstance();

        common::lock_guard_t lock(devMngr.deviceMutex);
        unsigned nDevices = 0;
        for (auto& device : devMngr.mDevices) {
            // const Platform platform(device->getInfo<CL_DEVICE_PLATFORM>());

            string dstr = device->get_info<sycl::info::device::name>();
            bool show_braces =
                (static_cast<unsigned>(getActiveDeviceId()) == nDevices);

            string id = (show_braces ? string("[") : "-") +
                        to_string(nDevices) + (show_braces ? string("]") : "-");
            size_t msize =
                device->get_info<sycl::info::device::global_mem_size>();
            info << id << " " << getPlatformName(*device) << ": " << ltrim(dstr)
                 << ", " << msize / 1048576 << " MB";
            info << " (";
            if (device->has(aspect::fp64)) { info << "fp64 "; }
            if (device->has(aspect::fp16)) { info << "fp16 "; }
            info << "\b)";
#ifndef NDEBUG
            info << " -- ";
            string devVersion = device->get_info<sycl::info::device::version>();
            string driVersion =
                device->get_info<sycl::info::device::driver_version>();
            info << devVersion;
            info << " -- Device driver " << driVersion;
            info << " -- Unified Memory ("
                 << (isHostUnifiedMemory(*device) ? "True" : "False") << ")";
#endif
            info << endl;

            nDevices++;
        }
    } catch (const AfError& err) {
        UNUSED(err);
        info << "No platforms found.\n";
        // Don't throw an exception here. Info should pass even if the system
        // doesn't have the correct drivers installed.
    }
    return info.str();
}

string getPlatformName(const sycl::device& device) {
    std::string platStr =
        device.get_platform().get_info<sycl::info::platform::name>();
    // return platformMap(platStr);
    return platStr;
}

typedef pair<unsigned, unsigned> device_id_t;

pair<unsigned, unsigned>& tlocalActiveDeviceId() {
    // First element is active context id
    // Second element is active queue id
    thread_local device_id_t activeDeviceId(0, 0);

    return activeDeviceId;
}

void setActiveContext(int device) {
    tlocalActiveDeviceId() = make_pair(device, device);
}

int getDeviceCount() noexcept try {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);
    return static_cast<int>(devMngr.mQueues.size());
} catch (const AfError& err) {
    UNUSED(err);
    // If device manager threw an error then return 0 because no platforms
    // were found
    return 0;
}

void init() {
    thread_local const DeviceManager& devMngr = DeviceManager::getInstance();
    UNUSED(devMngr);
}

unsigned getActiveDeviceId() {
    // Second element is the queue id, which is
    // what we mean by active device id in opencl backend
    return get<1>(tlocalActiveDeviceId());
}

/*
int getDeviceIdFromNativeId(cl_device_id id) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    int nDevices = static_cast<int>(devMngr.mDevices.size());
    int devId    = 0;
    for (devId = 0; devId < nDevices; ++devId) {
        //TODO: how to get cl_device_id from sycl::device
        if (id == devMngr.mDevices[devId]->get()) { return devId; }
    }
    // TODO: reasonable if no match??
    return -1;
}
*/

int getActiveDeviceType() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return devMngr.mDeviceTypes[get<1>(devId)];
}

int getActivePlatform() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return devMngr.mPlatforms[get<1>(devId)];
}

const sycl::context& getContext() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return *(devMngr.mContexts[get<0>(devId)]);
}

sycl::queue& getQueue() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return *(devMngr.mQueues[get<1>(devId)]);
}

sycl::queue* getQueueHandle(int device_id) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return devMngr.mQueues[device_id].get();
}

const sycl::device& getDevice(int id) {
    device_id_t& devId = tlocalActiveDeviceId();

    if (id == -1) { id = get<1>(devId); }

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);
    return *(devMngr.mDevices[id]);
}

const std::string& getActiveDeviceBaseBuildFlags() {
    device_id_t& devId     = tlocalActiveDeviceId();
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);
    return devMngr.mBaseOpenCLBuildFlags[get<1>(devId)];
}

size_t getDeviceMemorySize(int device) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);
    // Assuming devices don't deallocate or are invalidated during execution
    sycl::device& dev = *devMngr.mDevices[device];
    return dev.get_info<sycl::info::device::global_mem_size>();
}

size_t getHostMemorySize() { return common::getHostMemorySize(); }

sycl::info::device_type getDeviceType() {
    const sycl::device& device = getDevice();
    sycl::info::device_type type =
        device.get_info<sycl::info::device::device_type>();
    return type;
}

bool isHostUnifiedMemory(const sycl::device& device) {
    return device.has(sycl::aspect::usm_host_allocations);
}

bool OneAPICPUOffload(bool forceOffloadOSX) {
    static const bool offloadEnv = getEnvVar("AF_ONEAPI_CPU_OFFLOAD") != "0";
    bool offload                 = false;
    if (offloadEnv) { offload = isHostUnifiedMemory(getDevice()); }
#if OS_MAC
    // FORCED OFFLOAD FOR LAPACK FUNCTIONS ON OSX UNIFIED MEMORY DEVICES
    //
    // On OSX Unified Memory devices (Intel), always offload LAPACK but not GEMM
    // irrespective of the AF_OPENCL_CPU_OFFLOAD value
    // From GEMM, OpenCLCPUOffload(false) is called which will render the
    // variable inconsequential to the returned result.
    //
    // Issue https://github.com/arrayfire/arrayfire/issues/662
    //
    // Make sure device has unified memory
    bool osx_offload = isHostUnifiedMemory(getDevice());
    // Force condition
    offload = osx_offload && (offload || forceOffloadOSX);
#else
    UNUSED(forceOffloadOSX);
#endif
    return offload;
}

bool isGLSharingSupported() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return devMngr.mIsGLSharingOn[get<1>(devId)];
}

bool isDoubleSupported(unsigned device) {
    DeviceManager& devMngr = DeviceManager::getInstance();
    {
        common::lock_guard_t lock(devMngr.deviceMutex);
        sycl::device& dev = *devMngr.mDevices[device];
        return dev.has(sycl::aspect::fp64);
    }
}

bool isHalfSupported(unsigned device) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);
    return devMngr.mDevices[device]->has(sycl::aspect::fp16);
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute) {
    ONEAPI_NOT_SUPPORTED("");
}

int setDevice(int device) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    if (device >= static_cast<int>(devMngr.mQueues.size()) ||
        device >= static_cast<int>(DeviceManager::MAX_DEVICES)) {
        return -1;
    } else {
        int old = getActiveDeviceId();
        setActiveContext(device);
        return old;
    }
}

void sync(int device) {
    int currDevice = getActiveDeviceId();
    setDevice(device);
    getQueue().wait();
    setDevice(currDevice);
}

void addDeviceContext(sycl::device& dev, sycl::context& ctx, sycl::queue& que) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    int nDevices = 0;
    {
        common::lock_guard_t lock(devMngr.deviceMutex);

        auto tDevice  = make_unique<sycl::device>(dev);
        auto tContext = make_unique<sycl::context>(ctx);
        // queue atleast has implicit context and device if created
        auto tQueue = make_unique<sycl::queue>(que);

        devMngr.mPlatforms.push_back(getPlatformEnum(*tDevice));
        // FIXME: add OpenGL Interop for user provided contexts later
        devMngr.mIsGLSharingOn.push_back(false);
        devMngr.mDeviceTypes.push_back(static_cast<int>(
            tDevice->get_info<sycl::info::device::device_type>()));

        devMngr.mDevices.push_back(move(tDevice));
        devMngr.mContexts.push_back(move(tContext));
        devMngr.mQueues.push_back(move(tQueue));
        nDevices = static_cast<int>(devMngr.mDevices.size()) - 1;

        // TODO: cache?
    }

    // Last/newly added device needs memory management
    memoryManager().addMemoryManagement(nDevices);
}

void setDeviceContext(sycl::device& dev, sycl::context& ctx) {
    // FIXME: add OpenGL Interop for user provided contexts later
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    const int dCount = static_cast<int>(devMngr.mDevices.size());
    for (int i = 0; i < dCount; ++i) {
        if (*devMngr.mDevices[i] == dev && *devMngr.mContexts[i] == ctx) {
            setActiveContext(i);
            return;
        }
    }
    AF_ERROR("No matching device found", AF_ERR_ARG);
}

void removeDeviceContext(sycl::device& dev, sycl::context& ctx) {
    if (getDevice() == dev && getContext() == ctx) {
        AF_ERROR("Cannot pop the device currently in use", AF_ERR_ARG);
    }

    DeviceManager& devMngr = DeviceManager::getInstance();

    int deleteIdx = -1;
    {
        common::lock_guard_t lock(devMngr.deviceMutex);

        const int dCount = static_cast<int>(devMngr.mDevices.size());
        for (int i = 0; i < dCount; ++i) {
            if (*devMngr.mDevices[i] == dev && *devMngr.mContexts[i] == ctx) {
                deleteIdx = i;
                break;
            }
        }
    }

    if (deleteIdx < static_cast<int>(devMngr.mUserDeviceOffset)) {
        AF_ERROR("Cannot pop ArrayFire internal devices", AF_ERR_ARG);
    } else if (deleteIdx == -1) {
        AF_ERROR("No matching device found", AF_ERR_ARG);
    } else {
        // remove memory management for device added by user outside of the lock
        memoryManager().removeMemoryManagement(deleteIdx);

        common::lock_guard_t lock(devMngr.deviceMutex);
        // FIXME: this case can potentially cause issues due to the
        // modification of the device pool stl containers.

        // IF the current active device is enumerated at a position
        // that lies ahead of the device that has been requested
        // to be removed. We just pop the entries from pool since it
        // has no side effects.
        devMngr.mDevices.erase(devMngr.mDevices.begin() + deleteIdx);
        devMngr.mContexts.erase(devMngr.mContexts.begin() + deleteIdx);
        devMngr.mQueues.erase(devMngr.mQueues.begin() + deleteIdx);
        devMngr.mPlatforms.erase(devMngr.mPlatforms.begin() + deleteIdx);

        // FIXME: add OpenGL Interop for user provided contexts later
        devMngr.mIsGLSharingOn.erase(devMngr.mIsGLSharingOn.begin() +
                                     deleteIdx);

        // OTHERWISE, update(decrement) the thread local active device ids
        device_id_t& devId = tlocalActiveDeviceId();

        if (deleteIdx < static_cast<int>(devId.first)) {
            device_id_t newVals = make_pair(devId.first - 1, devId.second - 1);
            devId               = newVals;
        }
    }
}

unsigned getMemoryBusWidth(const sycl::device& device) {
    return device.get_info<sycl::info::device::global_mem_cache_line_size>();
}

size_t getL2CacheSize(const sycl::device& device) {
    return device.get_info<sycl::info::device::global_mem_cache_line_size>();
}

unsigned getComputeUnits(const sycl::device& device) {
    return device.get_info<sycl::info::device::max_compute_units>();
}

unsigned getMaxParallelThreads(const sycl::device& device) {
    return getComputeUnits(device) * 2048;
}

bool synchronize_calls() {
    static const bool sync = getEnvVar("AF_SYNCHRONOUS_CALLS") == "1";
    return sync;
}

int& getMaxJitSize() {
#if defined(OS_MAC)
    constexpr int MAX_JIT_LEN = 50;
#else
    constexpr int MAX_JIT_LEN = 100;
#endif
    thread_local int length = 0;
    if (length <= 0) {
        string env_var = getEnvVar("AF_OPENCL_MAX_JIT_LEN");
        if (!env_var.empty()) {
            int input_len = stoi(env_var);
            length        = input_len > 0 ? input_len : MAX_JIT_LEN;
        } else {
            length = MAX_JIT_LEN;
        }
    }
    return length;
}

bool& evalFlag() {
    thread_local bool flag = true;
    return flag;
}

MemoryManagerBase& memoryManager() {
    static once_flag flag;

    DeviceManager& inst = DeviceManager::getInstance();

    call_once(flag, [&]() {
        // By default, create an instance of the default memory manager
        inst.memManager = make_unique<common::DefaultMemoryManager>(
            getDeviceCount(), common::MAX_BUFFERS,
            AF_MEM_DEBUG || AF_ONEAPI_MEM_DEBUG);
        // Set the memory manager's device memory manager
        unique_ptr<Allocator> deviceMemoryManager;
        deviceMemoryManager = make_unique<Allocator>();
        inst.memManager->setAllocator(move(deviceMemoryManager));
        inst.memManager->initialize();
    });

    return *(inst.memManager.get());
}

MemoryManagerBase& pinnedMemoryManager() {
    static once_flag flag;

    DeviceManager& inst = DeviceManager::getInstance();

    call_once(flag, [&]() {
        // By default, create an instance of the default memory manager
        inst.pinnedMemManager = make_unique<common::DefaultMemoryManager>(
            getDeviceCount(), common::MAX_BUFFERS,
            AF_MEM_DEBUG || AF_ONEAPI_MEM_DEBUG);
        // Set the memory manager's device memory manager
        unique_ptr<AllocatorPinned> deviceMemoryManager;
        deviceMemoryManager = make_unique<AllocatorPinned>();
        inst.pinnedMemManager->setAllocator(move(deviceMemoryManager));
        inst.pinnedMemManager->initialize();
    });

    return *(inst.pinnedMemManager.get());
}

void setMemoryManager(unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManager(move(mgr));
}

void resetMemoryManager() {
    return DeviceManager::getInstance().resetMemoryManager();
}

void setMemoryManagerPinned(unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManagerPinned(move(mgr));
}

void resetMemoryManagerPinned() {
    return DeviceManager::getInstance().resetMemoryManagerPinned();
}

arrayfire::common::ForgeManager& forgeManager() {
    return *(DeviceManager::getInstance().fgMngr);
}

GraphicsResourceManager& interopManager() {
    static once_flag initFlags[DeviceManager::MAX_DEVICES];

    int id = getActiveDeviceId();

    DeviceManager& inst = DeviceManager::getInstance();

    call_once(initFlags[id], [&] {
        inst.gfxManagers[id] = make_unique<GraphicsResourceManager>();
    });

    return *(inst.gfxManagers[id].get());
}

unique_ptr<PlanCache>& oneFFTManager(const int deviceId) {
    thread_local unique_ptr<PlanCache> caches[DeviceManager::MAX_DEVICES];
    thread_local once_flag initFlags[DeviceManager::MAX_DEVICES];
    call_once(initFlags[deviceId],
              [&] { caches[deviceId] = make_unique<PlanCache>(); });
    return caches[deviceId];
}

PlanCache& fftManager() { return *oneFFTManager(getActiveDeviceId()); }

}  // namespace oneapi
}  // namespace arrayfire

/*
//TODO: select which external api functions to expose and add to
header+implement

using namespace oneapi;

af_err afcl_get_device_type(afcl_device_type* res) {
    try {
        *res = static_cast<afcl_device_type>(getActiveDeviceType());
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_platform(afcl_platform* res) {
    try {
        *res = static_cast<afcl_platform>(getActivePlatform());
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_context(cl_context* ctx, const bool retain) {
    try {
        *ctx = getContext()();
        if (retain) { clRetainContext(*ctx); }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_queue(cl_command_queue* queue, const bool retain) {
    try {
        *queue = getQueue()();
        if (retain) { clRetainCommandQueue(*queue); }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_device_id(cl_device_id* id) {
    try {
        *id = getDevice()();
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_set_device_id(cl_device_id id) {
    try {
        setDevice(getDeviceIdFromNativeId(id));
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_add_device_context(cl_device_id dev, cl_context ctx,
                               cl_command_queue que) {
    try {
        addDeviceContext(dev, ctx, que);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_set_device_context(cl_device_id dev, cl_context ctx) {
    try {
        setDeviceContext(dev, ctx);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_delete_device_context(cl_device_id dev, cl_context ctx) {
    try {
        removeDeviceContext(dev, ctx);
    }
    CATCHALL;
    return AF_SUCCESS;
}
*/
