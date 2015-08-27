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

#include <cl.hpp>
#include <vector>
#include <string>

namespace opencl
{

class DeviceManager
{
    friend std::string getInfo();

    friend int getDeviceCount();

    friend int getActiveDeviceId();

    friend const cl::Context& getContext();

    friend cl::CommandQueue& getQueue();

    friend const cl::Device& getDevice();

    friend bool isGLSharingSupported();

    friend bool isDoubleSupported(int device);

    friend void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    friend int setDevice(int device);

    public:
        static const unsigned MAX_DEVICES = 16;

        static DeviceManager& getInstance();

        ~DeviceManager();

    protected:
        void setContext(int device);

        DeviceManager();

        // Following two declarations are required to
        // avoid copying accidental copy/assignment
        // of instance returned by getInstance to other
        // variables
        DeviceManager(DeviceManager const&);
        void operator=(DeviceManager const&);
#if defined(WITH_GRAPHICS)
        void markDeviceForInterop(const int device, const fg::Window* wHandle);
#endif

    private:
        // Attributes
        std::vector<cl::CommandQueue*>  mQueues;
        std::vector<cl::Device*>       mDevices;
        std::vector<cl::Context*>     mContexts;
        std::vector<cl::Platform*>   mPlatforms;
        std::vector<unsigned>       mCtxOffsets;
        std::vector<bool>        mIsGLSharingOn;

        unsigned mActiveCtxId;
        unsigned mActiveQId;
};

std::string getInfo();

int getDeviceCount();

int getActiveDeviceId();

const cl::Context& getContext();

cl::CommandQueue& getQueue();

const cl::Device& getDevice();

bool isGLSharingSupported();

bool isDoubleSupported(int device);

void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

std::string getPlatformName(const cl::Device &device);

int setDevice(int device);

void sync(int device);

}
