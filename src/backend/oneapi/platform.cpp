/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/graphics_common.hpp>
#include <GraphicsResourceManager.hpp>
#include <blas.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <err_oneapi.hpp>
#include <errorcodes.hpp>
#include <version.hpp>
#include <af/version.h>
#include <memory.hpp>

#ifdef OS_MAC
#include <OpenGL/CGLCurrent.h>
#endif

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

using sycl::queue;
using sycl::context;
using sycl::device;
using sycl::platform;
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

using common::memory::MemoryManagerBase;
using oneapi::Allocator;
using oneapi::AllocatorPinned;

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

/*
afcl::platform getPlatformEnum(cl::Device dev) {
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
*/

string getDeviceInfo() noexcept {
    ONEAPI_NOT_SUPPORTED("");
    return "";
}

string getPlatformName(const sycl::device& device) {
    ONEAPI_NOT_SUPPORTED("");
    return "";
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

int getDeviceCount() noexcept {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

void init() {
    ONEAPI_NOT_SUPPORTED("");
}

unsigned getActiveDeviceId() {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

/*
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
*/

int getActiveDeviceType() {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

int getActivePlatform() {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}
const context& getContext() {
    ONEAPI_NOT_SUPPORTED("");
    sycl::context c;
    return c;
    /*
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return *(devMngr.mContexts[get<0>(devId)]);
    */
}

sycl::queue& getQueue() {
    sycl::queue q;
    return q; 
    /*
    device_id_t& devId = tlocalActiveDeviceId();

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);

    return *(devMngr.mQueues[get<1>(devId)]);
    */
}

const sycl::device& getDevice(int id) {
    sycl::device d;
    return d;
    /*
    device_id_t& devId = tlocalActiveDeviceId();

    if (id == -1) { id = get<1>(devId); }

    DeviceManager& devMngr = DeviceManager::getInstance();

    common::lock_guard_t lock(devMngr.deviceMutex);
    return *(devMngr.mDevices[id]);
    */
}

size_t getDeviceMemorySize(int device) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

size_t getHostMemorySize() { return common::getHostMemorySize(); }

/*
cl_device_type getDeviceType() {
    const sycl::device& device = getDevice();
    cl_device_type type        = device.getInfo<CL_DEVICE_TYPE>();
    return type;
}
*/

bool isHostUnifiedMemory(const sycl::device& device) {
    ONEAPI_NOT_SUPPORTED("");
    return false;
}

bool OpenCLCPUOffload(bool forceOffloadOSX) {
    ONEAPI_NOT_SUPPORTED("");
    return false;
}

bool isGLSharingSupported() {
    ONEAPI_NOT_SUPPORTED("");
    return false;
}

bool isDoubleSupported(unsigned device) {
    ONEAPI_NOT_SUPPORTED("");
    return false;
}

bool isHalfSupported(unsigned device) {
    ONEAPI_NOT_SUPPORTED("");
    return false;
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute) {
    ONEAPI_NOT_SUPPORTED("");
}

int setDevice(int device) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

void sync(int device) {
    ONEAPI_NOT_SUPPORTED("");
}

void addDeviceContext(sycl::device dev, sycl::context ctx, sycl::queue que) {
    ONEAPI_NOT_SUPPORTED("");
}

void setDeviceContext(sycl::device dev, sycl::context ctx) {
    ONEAPI_NOT_SUPPORTED("");
}

void removeDeviceContext(sycl::device dev, sycl::context ctx) {
    ONEAPI_NOT_SUPPORTED("");
}

bool synchronize_calls() {
    return false;
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
    ONEAPI_NOT_SUPPORTED("");
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

/*
MemoryManagerBase& pinnedMemoryManager() {
    ONEAPI_NOT_SUPPORTED("");
}
*/

void setMemoryManager(unique_ptr<MemoryManagerBase> mgr) {
    ONEAPI_NOT_SUPPORTED("");
}

void resetMemoryManager() {
    ONEAPI_NOT_SUPPORTED("");
}

void setMemoryManagerPinned(unique_ptr<MemoryManagerBase> mgr) {
    ONEAPI_NOT_SUPPORTED("");
}

void resetMemoryManagerPinned() {
    ONEAPI_NOT_SUPPORTED("");
}

graphics::ForgeManager& forgeManager() {
    ONEAPI_NOT_SUPPORTED("");
}

GraphicsResourceManager& interopManager() {
    ONEAPI_NOT_SUPPORTED("");
}

}  // namespace oneapi

/*
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