/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/graphics_common.hpp>
#include <device_manager.hpp>
#include <memory.hpp>
#include <af/version.h>

#include <cctype>
#include <sstream>

using common::memory::MemoryManagerBase;
using std::string;

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
    : mVendorId("Unknown")
    , mModelName("Unknown")
    , mNumSMT(1)
    , mNumCores(1)
    , mNumLogCpus(1)
    , mIsHTT(false) {}

#endif

namespace cpu {

DeviceManager::DeviceManager()
    : queues(MAX_QUEUES), fgMngr(new graphics::ForgeManager()) {
    std::unique_ptr<MemoryManager> deviceMemoryManager;
    deviceMemoryManager.reset(new MemoryManager());
    memManager = std::move(deviceMemoryManager);
}

DeviceManager& DeviceManager::getInstance() {
    static DeviceManager* my_instance = new DeviceManager();
    return *my_instance;
}

CPUInfo DeviceManager::getCPUInfo() const { return cinfo; }

void DeviceManager::setMemoryManager(std::unique_ptr<MemoryManagerBase> ptr) {
    // Get the current memory manager to prevent deadlock if it hasn't been
    // initialized yet
    memoryManager();
    memManager = std::move(ptr);
}

}  // namespace cpu
