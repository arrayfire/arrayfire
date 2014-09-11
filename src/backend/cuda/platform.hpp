#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace cuda
{

std::string getInfo();

std::string getDeviceInfo(int device);

std::string getPlatformInfo();

std::string getDriverVersion();

std::string getCUDARuntimeVersion();

std::string getInfo();

int getDeviceCount();

int getActiveDeviceId();

int getDeviceNativeId(int device);

int setDevice(int device);

cudaDeviceProp getDeviceProp(int device);

struct cudaDevice_t {
    cudaDeviceProp prop;
    size_t flops;
    int nativeId;

    //cudaDevice_t(cudaDeviceProp p, size_t f, int nId)
    //    : prop(p), flops(f), nativeId(nId)
    //{
    //}
};

class DeviceManager
{
    public:
        static const unsigned MAX_DEVICES = 16;

        static DeviceManager& getInstance();

        friend std::string getDeviceInfo(int device);

        friend std::string getPlatformInfo();

        friend std::string getDriverVersion();

        friend std::string getCUDARuntimeVersion();

        friend std::string getInfo();

        friend int getDeviceCount();

        friend int getActiveDeviceId();

        friend int getDeviceNativeId(int device);

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

        int setActiveDevice(int device);

        int activeDev;
        int nDevices;
};

}
