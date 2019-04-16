/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>
#include <common/graphics_common.hpp>
#include <common/host_memory.hpp>
#include <platform.hpp>
#include <version.hpp>
#include <af/version.h>

#include <cctype>
#include <sstream>

using namespace std;

#ifdef CPUID_CAPABLE

CPUInfo::CPUInfo()
    : mVendorId("")
    , mModelName("")
    , mNumSMT(0)
    , mNumCores(0)
    , mNumLogCpus(0)
    , mIsHTT(false) {
    // Get vendor name EAX=0
    CPUID cpuID1(1, 0);
    mIsHTT = cpuID1.EDX() & HTT_POS;

    CPUID cpuID0(0, 0);
    uint32_t HFS = cpuID0.EAX();
    mVendorId += string((const char*)&cpuID0.EBX(), 4);
    mVendorId += string((const char*)&cpuID0.EDX(), 4);
    mVendorId += string((const char*)&cpuID0.ECX(), 4);

    string upVId = mVendorId;

    for_each(upVId.begin(), upVId.end(), [](char& in) { in = ::toupper(in); });

    // Get num of cores
    if (upVId.find("INTEL") != std::string::npos) {
        mVendorId = "Intel";
        if (HFS >= 11) {
            for (int lvl = 0; lvl < MAX_INTEL_TOP_LVL; ++lvl) {
                CPUID cpuID4(0x0B, lvl);
                uint32_t currLevel = (LVL_TYPE & cpuID4.ECX()) >> 8;
                switch (currLevel) {
                    case 0x01: mNumSMT = LVL_CORES & cpuID4.EBX(); break;
                    case 0x02: mNumLogCpus = LVL_CORES & cpuID4.EBX(); break;
                    default: break;
                }
            }
            // Fixes Possible divide by zero error
            // TODO: Fix properly
            mNumCores = mNumLogCpus / (mNumSMT == 0 ? 1 : mNumSMT);
        } else {
            if (HFS >= 1) {
                mNumLogCpus = (cpuID1.EBX() >> 16) & 0xFF;
                if (HFS >= 4) {
                    mNumCores = 1 + ((CPUID(4, 0).EAX() >> 26) & 0x3F);
                }
            }
            if (mIsHTT) {
                if (!(mNumCores > 1)) {
                    mNumCores   = 1;
                    mNumLogCpus = (mNumLogCpus >= 2 ? mNumLogCpus : 2);
                }
            } else {
                mNumCores = mNumLogCpus = 1;
            }
        }
    } else if (upVId.find("AMD") != std::string::npos) {
        mVendorId = "AMD";
        if (HFS >= 1) {
            mNumLogCpus = (cpuID1.EBX() >> 16) & 0xFF;
            if (CPUID(0x80000000, 0).EAX() >= 8) {
                mNumCores = 1 + ((CPUID(0x80000008, 0).ECX() & 0xFF));
            }
        }
        if (mIsHTT) {
            if (!(mNumCores > 1)) {
                mNumCores   = 1;
                mNumLogCpus = (mNumLogCpus >= 2 ? mNumLogCpus : 2);
            }
        } else {
            mNumCores = mNumLogCpus = 1;
        }
    } else {
        mVendorId = "Unknown";
    }
    // Get processor brand string
    // This seems to be working for both Intel & AMD vendors
    for (unsigned i = 0x80000002; i < 0x80000005; ++i) {
        CPUID cpuID(i, 0);
        mModelName += string((const char*)&cpuID.EAX(), 4);
        mModelName += string((const char*)&cpuID.EBX(), 4);
        mModelName += string((const char*)&cpuID.ECX(), 4);
        mModelName += string((const char*)&cpuID.EDX(), 4);
    }
    mModelName = string(mModelName.c_str());
}

#else

CPUInfo::CPUInfo()
    : mVendorId("")
    , mModelName("")
    , mNumSMT(0)
    , mNumCores(0)
    , mNumLogCpus(0)
    , mIsHTT(false) {
    mVendorId   = "Unknown";
    mModelName  = "Unknown";
    mNumSMT     = 1;
    mNumCores   = 1;
    mNumLogCpus = 1;
}

#endif

namespace cpu {

static const std::string get_system(void) {
    std::string arch = (sizeof(void*) == 4) ? "32-bit " : "64-bit ";

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
static inline std::string& ltrim(std::string& s) {
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

int getBackend() { return AF_BACKEND_CPU; }

std::string getDeviceInfo() {
    const CPUInfo cinfo = DeviceManager::getInstance().getCPUInfo();

    std::ostringstream info;

    info << "ArrayFire v" << AF_VERSION << " (CPU, " << get_system()
         << ", build " << AF_REVISION << ")" << std::endl;

    std::string model = cinfo.model();

    size_t memMB = getDeviceMemorySize(getActiveDeviceId()) / 1048576;

    info << string("[0] ") << cinfo.vendor() << ": " << ltrim(model);

    if (memMB)
        info << ", " << memMB << " MB, ";
    else
        info << ", Unknown MB, ";

    info << "Max threads(" << cinfo.threads() << ") ";
#ifndef NDEBUG
    info << AF_COMPILER_STR;
#endif
    info << std::endl;

    return info.str();
}

bool isDoubleSupported(int device) {
    UNUSED(device);
    return DeviceManager::IS_DOUBLE_SUPPORTED;
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute) {
    const CPUInfo cinfo = DeviceManager::getInstance().getCPUInfo();

    snprintf(d_name, 64, "%s", cinfo.vendor().c_str());
    snprintf(d_platform, 10, "CPU");
    // report the compiler for toolkit
    snprintf(d_toolkit, 64, "%s", AF_COMPILER_STR);
    snprintf(d_compute, 10, "%s", "0.0");
}

unsigned getMaxJitSize() {
    const int MAX_JIT_LEN = 100;

    thread_local int length = 0;
    if (length == 0) {
        std::string env_var = getEnvVar("AF_CPU_MAX_JIT_LEN");
        if (!env_var.empty()) {
            length = std::stoi(env_var);
        } else {
            length = MAX_JIT_LEN;
        }
    }
    return length;
}

int getDeviceCount() { return DeviceManager::NUM_DEVICES; }

// Get the currently active device id
int getActiveDeviceId() { return DeviceManager::ACTIVE_DEVICE_ID; }

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

CPUInfo DeviceManager::getCPUInfo() const { return cinfo; }

void sync(int device) { getQueue(device).sync(); }

bool& evalFlag() {
    thread_local bool flag = true;
    return flag;
}

DeviceManager::DeviceManager()
    : queues(MAX_QUEUES)
    , memManager(new MemoryManager())
    , fgMngr(new graphics::ForgeManager()) {}

MemoryManager& memoryManager() {
    DeviceManager& inst = DeviceManager::getInstance();
    return *(inst.memManager);
}

graphics::ForgeManager& forgeManager() {
    return *(DeviceManager::getInstance().fgMngr);
}

DeviceManager& DeviceManager::getInstance() {
    static DeviceManager* my_instance = new DeviceManager();
    return *my_instance;
}

}  // namespace cpu
