/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#if defined(WITH_GRAPHICS)
#include <fg/window.h>
#endif

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <CL/cl2.hpp>
#pragma GCC diagnostic pop

#include <memory>
#include <vector>
#include <string>

#include <cache.hpp>
#include <memory.hpp>
#include <GraphicsResourceManager.hpp>
#include <clfft.hpp>
#include <common/types.hpp>

namespace opencl
{
int getBackend();

std::string getDeviceInfo();

int getDeviceCount();

int getActiveDeviceId();

unsigned getMaxJitSize();

const cl::Context& getContext();

cl::CommandQueue& getQueue();

const cl::Device& getDevice(int id = -1);

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

cl_device_type getDeviceType();

bool isHostUnifiedMemory(const cl::Device &device);

bool OpenCLCPUOffload(bool forceOffloadOSX = true);

bool isGLSharingSupported();

bool isDoubleSupported(int device);

void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

std::string getPlatformName(const cl::Device &device);

int setDevice(int device);

void addDeviceContext(cl_device_id dev, cl_context cxt, cl_command_queue que);

void setDeviceContext(cl_device_id dev, cl_context cxt);

void removeDeviceContext(cl_device_id dev, cl_context ctx);

void sync(int device);

bool synchronize_calls();

int getActiveDeviceType();
int getActivePlatform();

bool& evalFlag();

///////////////////////// BEGIN Sub-Managers ///////////////////
//
MemoryManager& memoryManager();

MemoryManagerPinned& pinnedMemoryManager();

#if defined(WITH_GRAPHICS)
GraphicsResourceManager& interopManager();
#endif

PlanCache& fftManager();

void addKernelToCache(int device, const std::string& key, const kc_entry_t entry);

void removeKernelFromCache(int device, const std::string& key);

kc_entry_t kernelCache(int device, const std::string& key);
//
///////////////////////// END Sub-Managers /////////////////////

class DeviceManager
{
    friend MemoryManager& memoryManager();

    friend MemoryManagerPinned& pinnedMemoryManager();

#if defined(WITH_GRAPHICS)
    friend GraphicsResourceManager& interopManager();
#endif

    friend PlanCache& fftManager();

    friend void addKernelToCache(int device, const std::string& key, const kc_entry_t entry);

    friend void removeKernelFromCache(int device, const std::string& key);

    friend kc_entry_t kernelCache(int device, const std::string& key);

    friend std::string getDeviceInfo();

    friend int getDeviceCount();

    friend int getDeviceIdFromNativeId(cl_device_id id);

    friend const cl::Context& getContext();

    friend cl::CommandQueue& getQueue();

    friend const cl::Device& getDevice(int id);

    friend size_t getDeviceMemorySize(int device);

    friend bool isGLSharingSupported();

    friend bool isDoubleSupported(int device);

    friend void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    friend int setDevice(int device);

    friend void addDeviceContext(cl_device_id dev, cl_context cxt, cl_command_queue que);

    friend void setDeviceContext(cl_device_id dev, cl_context cxt);

    friend void removeDeviceContext(cl_device_id dev, cl_context ctx);

    friend int getActiveDeviceType();

    friend int getActivePlatform();

    public:
        static const unsigned MAX_DEVICES = 32;

        static DeviceManager& getInstance();

        ~DeviceManager();

    protected:
        DeviceManager();

        // Following two declarations are required to
        // avoid copying accidental copy/assignment
        // of instance returned by getInstance to other
        // variables
        DeviceManager(DeviceManager const&);
        void operator=(DeviceManager const&);
#if defined(WITH_GRAPHICS)
        void markDeviceForInterop(const int device, const forge::Window* wHandle);
#endif

    private:
        // Attributes
        common::mutex_t deviceMutex;
        std::vector<cl::Device*>       mDevices;
        std::vector<cl::Context*>     mContexts;
        std::vector<cl::CommandQueue*>  mQueues;
        std::vector<bool>        mIsGLSharingOn;
        std::vector<int>         mDeviceTypes;
        std::vector<int>         mPlatforms;
        unsigned mUserDeviceOffset;

        std::unique_ptr<MemoryManager> memManager;
        std::unique_ptr<MemoryManagerPinned> pinnedMemManager;

#if defined(WITH_GRAPHICS)
        std::unique_ptr<GraphicsResourceManager> gfxManagers[MAX_DEVICES];
#endif
        clfftSetupData mFFTSetup;
};
}
