/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#if defined(WITH_GRAPHICS)
#include <fg/window.h>
#endif

namespace cuda
{

int getBackend();

std::string getDeviceInfo();
std::string getDeviceInfo(int device);

std::string getPlatformInfo();

std::string getDriverVersion();

std::string getCUDARuntimeVersion();

bool isDoubleSupported(int device);

void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

unsigned getMaxJitSize();

int getDeviceCount();

int getActiveDeviceId();

int getDeviceNativeId(int device);

cudaStream_t getStream(int device);

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

int setDevice(int device);

void sync(int device);

// Returns true if the AF_SYNCHRONIZE_CALLS environment variable is set to 1
bool synchronize_calls();

cudaDeviceProp getDeviceProp(int device);

struct cudaDevice_t {
    cudaDeviceProp prop;
    size_t flops;
    int nativeId;
};

bool& evalFlag();

class DeviceManager
{
    public:
        static const unsigned MAX_DEVICES = 16;

        static DeviceManager& getInstance();

        friend std::string getDeviceInfo(int device);

        friend std::string getPlatformInfo();

        friend std::string getDriverVersion();

        friend std::string getCUDARuntimeVersion();

        friend std::string getDeviceInfo();

        friend int getDeviceCount();

        friend int getActiveDeviceId();

        friend int getDeviceNativeId(int device);

        friend int getDeviceIdFromNativeId(int nativeId);

        friend cudaStream_t getStream(int device);

        friend int setDevice(int device);

        friend cudaDeviceProp getDeviceProp(int device);

    private:
        DeviceManager();

        // Following two declarations are required to
        // avoid copying accidental copy/assignment
        // of instance returned by getInstance to other
        // variables
        DeviceManager(DeviceManager const&);
        void operator=(DeviceManager const&);

        // Attributes
        std::vector<cudaDevice_t> cuDevices;

        enum sort_mode {flops = 0, memory = 1, compute = 2, none = 3};

        void sortDevices(sort_mode mode = flops);

        int setActiveDevice(int device, int native = -1);

        int activeDev;
        int nDevices;
        cudaStream_t streams[MAX_DEVICES];
};

}
