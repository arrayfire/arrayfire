/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Include this before af/opencl.h
// Causes conflict between system cl.hpp and opencl/cl.hpp
#include <common/graphics_common.hpp>

#include <GraphicsResourceManager.hpp>
#include <blas.hpp>
#include <build_version.hpp>
#include <clfft.hpp>
#include <common/ArrayFireTypesIO.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/Version.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <err_opencl.hpp>
#include <errorcodes.hpp>
#include <platform.hpp>
#include <af/version.h>

#ifdef OS_MAC
#include <OpenGL/CGLCurrent.h>
#endif

#include <boost/compute/context.hpp>
#include <boost/compute/utility/program_cache.hpp>

#include <cctype>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using cl::CommandQueue;
using cl::Context;
using cl::Device;
using cl::Platform;
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
using arrayfire::common::Version;
using arrayfire::opencl::Allocator;
using arrayfire::opencl::AllocatorPinned;

namespace arrayfire {
namespace opencl {

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

int getBackend() { return AF_BACKEND_OPENCL; }

bool verify_present(const string& pname, const string ref) {
    auto iter =
        search(begin(pname), end(pname), begin(ref), end(ref),
               [](const string::value_type& l, const string::value_type& r) {
                   return tolower(l) == tolower(r);
               });

    return iter != end(pname);
}

static string platformMap(string& platStr) {
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

afcl::platform getPlatformEnum(Device dev) {
    string pname = getPlatformName(dev);
    if (verify_present(pname, "AMD"))
        return AFCL_PLATFORM_AMD;
    else if (verify_present(pname, "NVIDIA"))
        return AFCL_PLATFORM_NVIDIA;
    else if (verify_present(pname, "INTEL"))
        return AFCL_PLATFORM_INTEL;
    else if (verify_present(pname, "APPLE"))
        return AFCL_PLATFORM_APPLE;
    else if (verify_present(pname, "BEIGNET"))
        return AFCL_PLATFORM_BEIGNET;
    else if (verify_present(pname, "POCL"))
        return AFCL_PLATFORM_POCL;
    return AFCL_PLATFORM_UNKNOWN;
}

string getDeviceInfo() noexcept {
    ostringstream info;
    info << "ArrayFire v" << AF_VERSION << " (OpenCL, " << get_system()
         << ", build " << AF_REVISION << ")\n";

    vector<cl::Device*> devices;
    try {
        DeviceManager& devMngr = DeviceManager::getInstance();

        common::lock_guard_t lock(devMngr.deviceMutex);
        unsigned nDevices = 0;
        for (auto& device : devMngr.mDevices) {
            const Platform platform(device->getInfo<CL_DEVICE_PLATFORM>());

            string dstr = device->getInfo<CL_DEVICE_NAME>();
            bool show_braces =
                (static_cast<unsigned>(getActiveDeviceId()) == nDevices);

            string id = (show_braces ? string("[") : "-") +
                        to_string(nDevices) + (show_braces ? string("]") : "-");

            size_t msize = device->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            info << id << " " << getPlatformName(*device) << ": " << ltrim(dstr)
                 << ", " << msize / 1048576 << " MB";
#ifndef NDEBUG
            info << " -- ";
            string devVersion = device->getInfo<CL_DEVICE_VERSION>();
            string driVersion = device->getInfo<CL_DRIVER_VERSION>();
            info << devVersion;
            info << " -- Device driver " << driVersion;
            info
                << " -- FP64 Support: "
                << (device->getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>() >
                            0
                        ? "True"
                        : "False");
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

string getPlatformName(const Device& device) {
    const Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
    string platStr = platform.getInfo<CL_PLATFORM_NAME>();
    return platformMap(platStr);
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

int getActiveDeviceId() {
    // Second element is the queue id, which is
    // what we mean by active device id in opencl backend
    return get<1>(tlocalActiveDeviceId());
}

int getDeviceIdFromNativeId(cl_device_id id) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    int nDevices = static_cast<int>(devMngr.mDevices.size());
    int devId    = 0;
    for (devId = 0; devId < nDevices; ++devId) {
        if (id == devMngr.mDevices[devId]->operator()()) { break; }
    }

    return devId;
}

int getActiveDeviceType() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return devMngr.mDeviceTypes[get<1>(devId)];
}

cl::Platform& getActivePlatform() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return *devMngr.mPlatforms[get<1>(devId)].first;
}

afcl::platform getActivePlatformVendor() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return devMngr.mPlatforms[get<1>(devId)].second;
}

bool isDeviceBufferAccessible(int buf_device_id, int execution_id) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return buf_device_id == execution_id ||
           *devMngr.mContexts[buf_device_id] ==
               *devMngr.mContexts[execution_id];
}

const Context& getContext() {
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return *(devMngr.mContexts[get<0>(devId)]);
}

cl_command_queue getQueueHandle(int device_id) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return (*(devMngr.mQueues[device_id]))();
}

CommandQueue& getQueue(int device_id) {
    device_id_t devId =
        (device_id = -1) ? tlocalActiveDeviceId()
                         : make_pair<unsigned, unsigned>(device_id, device_id);
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return *(devMngr.mQueues[get<1>(devId)]);
}

const Device& getDevice(int id) {
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
    return devMngr.mBaseBuildFlags[get<1>(devId)];
}

vector<Version> getOpenCLCDeviceVersion(const Device& device) {
    // For OpenCL-HPP >= v2023.12.14 type is cl::Platform instead of
    // cl_platform_id
    Platform device_platform;
    device_platform = device.getInfo<CL_DEVICE_PLATFORM>();

    auto platform_version = device_platform.getInfo<CL_PLATFORM_VERSION>();
    vector<Version> out;

    /// The ifdef allows us to support BUILDING ArrayFire with older
    /// versions of OpenCL where as the if condition in the ifdef allows us
    /// to support older versions of OpenCL at runtime
#ifdef CL_DEVICE_OPENCL_C_ALL_VERSIONS
    if (platform_version.substr(7).c_str()[0] >= '3') {
        vector<cl_name_version> device_versions =
            device.getInfo<CL_DEVICE_OPENCL_C_ALL_VERSIONS>();
        sort(begin(device_versions), end(device_versions),
             [](const auto& lhs, const auto& rhs) {
                 return lhs.version < rhs.version;
             });
        transform(begin(device_versions), end(device_versions),
                  std::back_inserter(out), [](const cl_name_version& version) {
                      return Version(CL_VERSION_MAJOR(version.version),
                                     CL_VERSION_MINOR(version.version),
                                     CL_VERSION_PATCH(version.version));
                  });
    } else {
#endif
        auto device_version = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
        int major           = atoi(device_version.substr(9, 1).c_str());
        int minor           = atoi(device_version.substr(11, 1).c_str());
        out.emplace_back(major, minor);
#ifdef CL_DEVICE_OPENCL_C_ALL_VERSIONS
    }
#endif
    return out;
}

size_t getDeviceMemorySize(int device) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    cl::Device dev;
    {
        common::lock_guard_t lock(devMngr.deviceMutex);
        // Assuming devices don't deallocate or are invalidated during execution
        dev = *devMngr.mDevices[device];
    }
    size_t msize = dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    return msize;
}

size_t getHostMemorySize() { return common::getHostMemorySize(); }

cl_device_type getDeviceType() {
    const cl::Device& device = getDevice();
    cl_device_type type      = device.getInfo<CL_DEVICE_TYPE>();
    return type;
}

bool OpenCLCPUOffload(bool forceOffloadOSX) {
    static const bool offloadEnv = getEnvVar("AF_OPENCL_CPU_OFFLOAD") != "0";
    bool offload                 = false;
    if (offloadEnv) { offload = getDeviceType() == CL_DEVICE_TYPE_CPU; }
#if OS_MAC
    // FORCED OFFLOAD FOR LAPACK FUNCTIONS ON OSX UNIFIED MEMORY DEVICES
    //
    // On OSX Unified Memory devices (Intel), always offload LAPACK but not GEMM
    // irrespective of the AF_OPENCL_CPU_OFFLOAD value
    // From GEMM, OpenCLCPUOffload(false) is called which will render the
    // variable inconsequential to the returned result.
    //
    // Issue https://github.com/arrayfire/arrayfire/issues/662
    // Force condition
    bool osx_offload = getDeviceType() == CL_DEVICE_TYPE_CPU;
    offload          = osx_offload && (offload || forceOffloadOSX);
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

    cl::Device dev;
    {
        common::lock_guard_t lock(devMngr.deviceMutex);
        dev = *devMngr.mDevices[device];
    }
    return isDoubleSupported(dev);
}

bool isHalfSupported(unsigned device) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    cl::Device dev;
    {
        common::lock_guard_t lock(devMngr.deviceMutex);
        dev = *devMngr.mDevices[device];
    }
    return isHalfSupported(dev);
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute) {
    unsigned nDevices    = 0;
    auto currActiveDevId = static_cast<unsigned>(getActiveDeviceId());
    bool devset          = false;

    DeviceManager& devMngr = DeviceManager::getInstance();

    {
        common::lock_guard_t lock(devMngr.deviceMutex);

        for (auto& context : devMngr.mContexts) {
            vector<Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();

            for (auto& device : devices) {
                const Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
                string platStr = platform.getInfo<CL_PLATFORM_NAME>();

                if (currActiveDevId == nDevices) {
                    string dev_str;
                    device.getInfo(CL_DEVICE_NAME, &dev_str);
                    string com_str = device.getInfo<CL_DEVICE_VERSION>();
                    com_str        = com_str.substr(7, 3);

                    // strip out whitespace from the device string:
                    const string& whitespace = " \t";
                    const auto strBegin = dev_str.find_first_not_of(whitespace);
                    const auto strEnd   = dev_str.find_last_not_of(whitespace);
                    const auto strRange = strEnd - strBegin + 1;
                    dev_str             = dev_str.substr(strBegin, strRange);

                    // copy to output
                    snprintf(d_name, 64, "%s", dev_str.c_str());
                    snprintf(d_platform, 10, "OpenCL");
                    snprintf(d_toolkit, 64, "%s", platStr.c_str());
                    snprintf(d_compute, 10, "%s", com_str.c_str());
                    devset = true;
                }
                if (devset) { break; }
                nDevices++;
            }
            if (devset) { break; }
        }
    }

    // Sanitize input
    for (int i = 0; i < 31; i++) {
        if (d_name[i] == ' ') {
            if (d_name[i + 1] == 0 || d_name[i + 1] == ' ') {
                d_name[i] = 0;
            } else {
                d_name[i] = '_';
            }
        }
    }
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
    getQueue().finish();
    setDevice(currDevice);
}

void addDeviceContext(cl_device_id dev, cl_context ctx, cl_command_queue que) {
    DeviceManager& devMngr = DeviceManager::getInstance();

    int nDevices = 0;
    {
        common::lock_guard_t lock(devMngr.deviceMutex);

        cl::Device tDevice(dev, true);
        cl::Context tContext(ctx, true);
        auto tQueue =
            (que == NULL ? make_unique<cl::CommandQueue>(tContext, tDevice)
                         : make_unique<cl::CommandQueue>(que, true));
        // FIXME: add OpenGL Interop for user provided contexts later
        devMngr.mIsGLSharingOn.push_back(false);
        devMngr.mDeviceTypes.push_back(
            static_cast<int>(tDevice.getInfo<CL_DEVICE_TYPE>()));

        // For OpenCL-HPP >= v2023.12.14 type is cl::Platform instead of
        // cl_platform_id
        cl::Platform device_platform;
        device_platform = tDevice.getInfo<CL_DEVICE_PLATFORM>();
        devMngr.mPlatforms.push_back(
            std::make_pair<std::unique_ptr<cl::Platform>, afcl_platform>(
                make_unique<cl::Platform>(device_platform(), true),
                getPlatformEnum(tDevice)));

        devMngr.mDevices.emplace_back(make_unique<cl::Device>(move(tDevice)));
        devMngr.mContexts.emplace_back(
            make_unique<cl::Context>(move(tContext)));
        devMngr.mQueues.push_back(move(tQueue));
        nDevices = static_cast<int>(devMngr.mDevices.size()) - 1;

        auto versions = getOpenCLCDeviceVersion(*(devMngr.mDevices.back()));
#ifdef AF_WITH_FAST_MATH
        std::string options =
            fmt::format(" -cl-std=CL{:Mm} -D dim_t={} -cl-fast-relaxed-math",
                        versions.back(), dtype_traits<dim_t>::getName());
#else
        std::string options =
            fmt::format(" -cl-std=CL{:Mm} -D dim_t={}", versions.back(),
                        dtype_traits<dim_t>::getName());
#endif
        devMngr.mBaseBuildFlags.push_back(options);

        // cache the boost program_cache object, clean up done on program exit
        // not during removeDeviceContext
        namespace compute = boost::compute;
        using BPCache     = DeviceManager::BoostProgCache;
        compute::context c(ctx);
        BPCache currCache = compute::program_cache::get_global_cache(c);
        devMngr.mBoostProgCacheVector.emplace_back(new BPCache(currCache));
    }

    // Last/newly added device needs memory management
    memoryManager().addMemoryManagement(nDevices);
}

void setDeviceContext(cl_device_id dev, cl_context ctx) {
    // FIXME: add OpenGL Interop for user provided contexts later
    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    const int dCount = static_cast<int>(devMngr.mDevices.size());
    for (int i = 0; i < dCount; ++i) {
        if (devMngr.mDevices[i]->operator()() == dev &&
            devMngr.mContexts[i]->operator()() == ctx) {
            setActiveContext(i);
            return;
        }
    }
    AF_ERROR("No matching device found", AF_ERR_ARG);
}

void removeDeviceContext(cl_device_id dev, cl_context ctx) {
    if (getDevice()() == dev && getContext()() == ctx) {
        AF_ERROR("Cannot pop the device currently in use", AF_ERR_ARG);
    }

    DeviceManager& devMngr = DeviceManager::getInstance();

    int deleteIdx = -1;
    {
        common::lock_guard_t lock(devMngr.deviceMutex);

        const int dCount = static_cast<int>(devMngr.mDevices.size());
        for (int i = static_cast<int>(devMngr.mUserDeviceOffset); i < dCount;
             ++i) {
            if (devMngr.mDevices[i]->operator()() == dev &&
                devMngr.mContexts[i]->operator()() == ctx) {
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
            AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG);
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
            AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG);
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

PlanCache& fftManager() {
    thread_local PlanCache clfftManagers[DeviceManager::MAX_DEVICES];

    return clfftManagers[getActiveDeviceId()];
}

}  // namespace opencl
}  // namespace arrayfire

using namespace arrayfire::opencl;

af_err afcl_get_device_type(afcl_device_type* res) {
    try {
        *res = static_cast<afcl_device_type>(getActiveDeviceType());
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_platform(afcl_platform* res) {
    try {
        *res = static_cast<afcl_platform>(getActivePlatformVendor());
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
