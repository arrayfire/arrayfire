/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/version.h>
#include <platform.hpp>
#include <sstream>
#include <queue.hpp>
#include <array>
#include <algorithm>
#include <iostream>
#include <string>
#include <defines.hpp>
#include <version.hpp>
#include <queue.hpp>
#include <host_memory.hpp>
#include <cctype>


#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(_WIN64)
#define CPUID_CAPABLE USE_CPUID
#else
#define CPUID_CAPABLE 0
#endif

#ifdef _WIN32
#include <limits.h>
#include <intrin.h>
typedef unsigned __int32  uint32_t;
#else
#include <stdint.h>
#endif

using namespace std;

#if CPUID_CAPABLE

#define MAX_INTEL_TOP_LVL 4

class CPUID {
    uint32_t regs[4];

    public:
    explicit CPUID(unsigned funcId, unsigned subFuncId) {
#ifdef _WIN32
        __cpuidex((int *)regs, (int)funcId, (int)subFuncId);

#else
        asm volatile
            ("cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
             : "a" (funcId), "c" (subFuncId));
#endif
    }

    inline const uint32_t &EAX() const { return regs[0]; }
    inline const uint32_t &EBX() const { return regs[1]; }
    inline const uint32_t &ECX() const { return regs[2]; }
    inline const uint32_t &EDX() const { return regs[3]; }
};

#endif

class CPUInfo {
    public:
        CPUInfo();
        string  vendor()   const { return mVendorId;   }
        string  model()    const { return mModelName;  }
        int     threads()  const { return mNumLogCpus; }

    private:
        // Bit positions for data extractions
        static const uint32_t LVL_NUM   = 0x000000FF;
        static const uint32_t LVL_TYPE  = 0x0000FF00;
        static const uint32_t LVL_CORES = 0x0000FFFF;
        static const uint32_t HTT_POS   = 0x10000000;

        // Attributes
        string mVendorId;
        string mModelName;
        int    mNumSMT;
        int    mNumCores;
        int    mNumLogCpus;
        bool   mIsHTT;
};

#if !CPUID_CAPABLE

CPUInfo::CPUInfo()
    : mVendorId(""), mModelName(""), mNumSMT(0), mNumCores(0), mNumLogCpus(0), mIsHTT(false)
{
    mVendorId = "Unknown";
    mModelName= "Unknown";
    mNumSMT   = 1;
    mNumCores = 1;
    mNumLogCpus = 1;
}

#else

CPUInfo::CPUInfo()
    : mVendorId(""), mModelName(""), mNumSMT(0), mNumCores(0), mNumLogCpus(0), mIsHTT(false)
{
    // Get vendor name EAX=0
    CPUID cpuID1(1, 0);
    mIsHTT   = cpuID1.EDX() & HTT_POS;

    CPUID cpuID0(0, 0);
    uint32_t HFS = cpuID0.EAX();
    mVendorId += string((const char *)&cpuID0.EBX(), 4);
    mVendorId += string((const char *)&cpuID0.EDX(), 4);
    mVendorId += string((const char *)&cpuID0.ECX(), 4);

    string upVId = mVendorId;
    for_each(upVId.begin(), upVId.end(), [](char& in) { in = ::toupper(in); });
    // Get num of cores
    if (upVId.find("INTEL") != std::string::npos) {
        mVendorId = "Intel";
        if(HFS >= 11) {
            for (int lvl=0; lvl<MAX_INTEL_TOP_LVL; ++lvl) {
                    CPUID cpuID4(0x0B, lvl);
                    uint32_t currLevel = (LVL_TYPE & cpuID4.ECX())>>8;
                    switch(currLevel) {
                        case 0x01: mNumSMT     = LVL_CORES & cpuID4.EBX(); break;
                        case 0x02: mNumLogCpus = LVL_CORES & cpuID4.EBX(); break;
                        default: break;
                    }
            }
            // Fixes Possible divide by zero error
            // TODO: Fix properly
            mNumCores = mNumLogCpus/(mNumSMT == 0 ? 1 : mNumSMT);
        } else {
            if (HFS>=1) {
                mNumLogCpus = (cpuID1.EBX() >> 16) & 0xFF;
                if (HFS>=4) {
                    mNumCores = 1 + ((CPUID(4, 0).EAX() >> 26) & 0x3F);
                }
            }
            if (mIsHTT) {
                if (!(mNumCores>1)) {
                    mNumCores = 1;
                    mNumLogCpus = (mNumLogCpus >= 2 ? mNumLogCpus : 2);
                }
            } else {
                mNumCores = mNumLogCpus = 1;
            }
        }
    } else if (upVId.find("AMD") != std::string::npos) {
        mVendorId = "AMD";
        if (HFS>=1) {
            mNumLogCpus = (cpuID1.EBX() >> 16) & 0xFF;
            if (CPUID(0x80000000, 0).EAX() >=8) {
                mNumCores = 1 + ((CPUID(0x80000008, 0).ECX() & 0xFF));
            }
        }
        if (mIsHTT) {
            if (!(mNumCores>1)) {
                mNumCores = 1;
                mNumLogCpus = (mNumLogCpus >= 2 ? mNumLogCpus : 2);
            }
        } else {
            mNumCores = mNumLogCpus = 1;
        }
    } else {
        mVendorId = "Unkown, probably ARM";
        cout<< "Unexpected vendor id" <<endl;
    }
    // Get processor brand string
    // This seems to be working for both Intel & AMD vendors
    for(unsigned i=0x80000002; i<0x80000005; ++i) {
        CPUID cpuID(i, 0);
        mModelName += string((const char*)&cpuID.EAX(), 4);
        mModelName += string((const char*)&cpuID.EBX(), 4);
        mModelName += string((const char*)&cpuID.ECX(), 4);
        mModelName += string((const char*)&cpuID.EDX(), 4);
    }
    mModelName = string(mModelName.c_str());
}

#endif

namespace cpu
{

unsigned getMaxJitSize()
{
    const int MAX_JIT_LEN = 20;

    static int length = 0;
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

int getBackend()
{
    return AF_BACKEND_CPU;
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

// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring/217605#217605
// trim from start
static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

std::string getDeviceInfo()
{
    std::ostringstream info;
    static CPUInfo cinfo;

    info << "ArrayFire v" << AF_VERSION
         << " (CPU, " << get_system() << ", build " << AF_REVISION << ")" << std::endl;
    std::string model = cinfo.model();
    size_t memMB = getDeviceMemorySize(getActiveDeviceId()) / 1048576;
    info << string("[0] ") << cinfo.vendor() <<": " << ltrim(model);

    if(memMB) info << ", " << memMB << " MB, ";
    else      info << ", Unknown MB, ";

    info << "Max threads("<< cinfo.threads()<<") ";
#ifndef NDEBUG
    info << AF_COMPILER_STR;
#endif
    info << std::endl;
    return info.str();
}

bool isDoubleSupported(int device)
{
    return true;
}

void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    static CPUInfo cinfo;
    snprintf(d_name, 64, "%s", cinfo.vendor().c_str());
    snprintf(d_platform, 10, "CPU");
    // report the compiler for toolkit
    snprintf(d_toolkit, 64, "%s", AF_COMPILER_STR);
    snprintf(d_compute, 10, "%s", "0.0");
}

int getDeviceCount()
{
    return 1;
}


int setDevice(int device)
{
    static bool flag;
    if(!flag && device != 0) {
        printf("WARNING af_set_device(device): device can only be 0 for CPU\n");
        flag = 1;
    }
    return 0;
}

int getActiveDeviceId()
{
    return 0;
}

size_t getDeviceMemorySize(int device)
{
    return common::getHostMemorySize();
}

size_t getHostMemorySize()
{
    return common::getHostMemorySize();
}

static const int MAX_QUEUES = 1;


queue& getQueue(int idx) {
    static std::array<queue, MAX_QUEUES> queues;
    return queues[idx];
}

void sync(int device)
{
    getQueue().sync();
}

bool& evalFlag()
{
    static bool flag = true;
    return flag;
}


}
