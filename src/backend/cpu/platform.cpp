/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/MemoryManagerBase.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <device_manager.hpp>
#include <platform.hpp>
#include <version.hpp>
#include <af/version.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>

using common::memory::MemoryManagerBase;
using std::endl;
using std::not1;
using std::ostringstream;
using std::ptr_fun;
using std::stoi;
using std::string;
using std::unique_ptr;

namespace cpu {

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

// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring/217605#217605
// trim from start
static inline string& ltrim(string& s) {
    s.erase(s.begin(),
            find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace))));
    return s;
}

int getBackend() { return AF_BACKEND_CPU; }

string getDeviceInfo() noexcept {
    const CPUInfo cinfo = DeviceManager::getInstance().getCPUInfo();

    ostringstream info;

    info << "ArrayFire v" << AF_VERSION << " (CPU, " << get_system()
         << ", build " << AF_REVISION << ")" << endl;

    string model = cinfo.model();

    size_t memMB =
        getDeviceMemorySize(static_cast<int>(getActiveDeviceId())) / 1048576;

    info << string("[0] ") << cinfo.vendor() << ": " << ltrim(model);

    if (memMB) {
        info << ", " << memMB << " MB, ";
    } else {
        info << ", Unknown MB, ";
    }

    info << "Max threads(" << cinfo.threads() << ") ";
#ifndef NDEBUG
    info << AF_COMPILER_STR;
#endif
    info << endl;

    return info.str();
}

bool isDoubleSupported(int device) {
    UNUSED(device);
    return DeviceManager::IS_DOUBLE_SUPPORTED;
}

bool isHalfSupported(int device) {
    UNUSED(device);
    return DeviceManager::IS_HALF_SUPPORTED;
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute) {
    const CPUInfo cinfo = DeviceManager::getInstance().getCPUInfo();

    snprintf(d_name, 64, "%s", cinfo.vendor().c_str());
    snprintf(d_platform, 10, "CPU");
    // report the compiler for toolkit
    snprintf(d_toolkit, 64, "%s", AF_COMPILER_STR);
    snprintf(d_compute, 10, "%s", "0.0");
}

int& getMaxJitSize() {
    constexpr int MAX_JIT_LEN = 100;
    thread_local int length   = 0;
    if (length <= 0) {
        string env_var = getEnvVar("AF_CPU_MAX_JIT_LEN");
        if (!env_var.empty()) {
            int input_len = std::stoi(env_var);
            length        = input_len > 0 ? input_len : MAX_JIT_LEN;
        } else {
            length = MAX_JIT_LEN;
        }
    }
    return length;
}

int getDeviceCount() { return DeviceManager::NUM_DEVICES; }

// Get the currently active device id
unsigned getActiveDeviceId() { return DeviceManager::ACTIVE_DEVICE_ID; }

size_t getDeviceMemorySize(int device) {
    UNUSED(device);
    return common::getHostMemorySize();
}

size_t getHostMemorySize() { return common::getHostMemorySize(); }

int setDevice(int device) {
    thread_local bool flag = false;
    if (!flag && device != 0) {
#ifndef NDEBUG
        fprintf(
            stderr,
            "WARNING af_set_device(device): device can only be 0 for CPU\n");
#endif
        flag = true;
    }
    return 0;
}

queue& getQueue(int device) {
    return DeviceManager::getInstance().queues[device];
}

void sync(int device) { getQueue(device).sync(); }

bool& evalFlag() {
    thread_local bool flag = true;
    return flag;
}

MemoryManagerBase& memoryManager() {
    DeviceManager& inst = DeviceManager::getInstance();
    return *(inst.memManager);
}

void setMemoryManager(unique_ptr<MemoryManagerBase> mgr) {
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

graphics::ForgeManager& forgeManager() {
    return *(DeviceManager::getInstance().fgMngr);
}

}  // namespace cpu
