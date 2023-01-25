/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/DefaultMemoryManager.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <device_manager.hpp>
#include <memory.hpp>
#include <af/version.h>

#include <cctype>
#include <sstream>

using arrayfire::common::MemoryManagerBase;
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
    mVendorId += string(reinterpret_cast<const char*>(&cpuID0.EBX()), 4);
    mVendorId += string(reinterpret_cast<const char*>(&cpuID0.EDX()), 4);
    mVendorId += string(reinterpret_cast<const char*>(&cpuID0.ECX()), 4);

    string upVId = mVendorId;

    for_each(upVId.begin(), upVId.end(),
             [](char& in) { in = static_cast<char>(::toupper(in)); });

    // Get num of cores
    if (upVId.find("INTEL") != std::string::npos) {
        mVendorId = "Intel";
        if (HFS >= 11) {
            for (int lvl = 0; lvl < MAX_INTEL_TOP_LVL; ++lvl) {
                CPUID cpuID4(0x0B, lvl);
                uint32_t currLevel = (LVL_TYPE & cpuID4.ECX()) >> 8U;
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
                mNumLogCpus = (cpuID1.EBX() >> 16U) & 0xFFU;
                if (HFS >= 4) {
                    mNumCores = 1 + ((CPUID(4, 0).EAX() >> 26U) & 0x3FU);
                }
            }
            if (mIsHTT) {
                if (!(mNumCores > 1)) {
                    mNumCores   = 1;
                    mNumLogCpus = (mNumLogCpus >= 2 ? mNumLogCpus : 2U);
                }
            } else {
                mNumCores = mNumLogCpus = 1;
            }
        }
    } else if (upVId.find("AMD") != std::string::npos) {
        mVendorId = "AMD";
        if (HFS >= 1) {
            mNumLogCpus = (cpuID1.EBX() >> 16U) & 0xFFU;
            if (CPUID(0x80000000, 0).EAX() >= 8U) {
                mNumCores = 1 + ((CPUID(0x80000008, 0).ECX() & 0xFFU));
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
        mModelName += string(reinterpret_cast<const char*>(&cpuID.EAX()), 4);
        mModelName += string(reinterpret_cast<const char*>(&cpuID.EBX()), 4);
        mModelName += string(reinterpret_cast<const char*>(&cpuID.ECX()), 4);
        mModelName += string(reinterpret_cast<const char*>(&cpuID.EDX()), 4);
    }
    mModelName.shrink_to_fit();
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

namespace arrayfire {
namespace cpu {

DeviceManager::DeviceManager()
    : queues(MAX_QUEUES)
    , fgMngr(new common::ForgeManager())
    , memManager(new common::DefaultMemoryManager(
          getDeviceCount(), common::MAX_BUFFERS,
          AF_MEM_DEBUG || AF_CPU_MEM_DEBUG)) {
    // Use the default ArrayFire memory manager
    std::unique_ptr<cpu::Allocator> deviceMemoryManager(new cpu::Allocator());
    memManager->setAllocator(std::move(deviceMemoryManager));
    memManager->initialize();
}

DeviceManager& DeviceManager::getInstance() {
    static auto* my_instance = new DeviceManager();
    return *my_instance;
}

CPUInfo DeviceManager::getCPUInfo() const { return cinfo; }

void DeviceManager::resetMemoryManager() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(
        new common::DefaultMemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                         AF_MEM_DEBUG || AF_CPU_MEM_DEBUG));
    setMemoryManager(std::move(mgr));
}

void DeviceManager::setMemoryManager(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a memory manager and the default memory
    // manager still hasn't been initialized, so initialize it anyways so we
    // don't inadvertently reset to it when we first call memoryManager()
    memoryManager();
    // Calls shutdown() on the existing memory manager
    if (memManager) { memManager->shutdownAllocator(); }
    memManager = std::move(newMgr);
    // Set the backend memory manager for this new manager to register native
    // functions correctly.
    std::unique_ptr<cpu::Allocator> deviceMemoryManager(new cpu::Allocator());
    memManager->setAllocator(std::move(deviceMemoryManager));
    memManager->initialize();
}

void DeviceManager::setMemoryManagerPinned(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    UNUSED(newMgr);
    UNUSED(this);
    AF_ERROR("Using pinned memory with CPU is not supported",
             AF_ERR_NOT_SUPPORTED);
}

void DeviceManager::resetMemoryManagerPinned() {
    // This is a NOOP - we should never set a pinned memory manager in the first
    // place for the CPU backend, but don't throw in case backend-agnostic
    // functions that operate on all memory managers need to call this
}

}  // namespace cpu
}  // namespace arrayfire
