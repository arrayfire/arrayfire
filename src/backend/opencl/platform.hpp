#pragma once

#include <cl.hpp>
#include <vector>
#include <string>

namespace opencl
{

std::string getInfo();

unsigned deviceCount();

unsigned getActiveDeviceId();

const cl::Context& getContext();

cl::CommandQueue& getQueue();

void setDevice(size_t device);

class DeviceManager
{
    friend std::string getInfo();

    friend unsigned deviceCount();

    friend unsigned getActiveDeviceId();

    friend const cl::Context& getContext();

    friend cl::CommandQueue& getQueue();

    friend void setDevice(size_t device);

    public:
        static const unsigned MAX_DEVICES = 16;

        static DeviceManager& getInstance();

    private:
        DeviceManager();

        // Following two declarations are required to
        // avoid copying accidental copy/assignment
        // of instance returned by getInstance to other
        // variables
        DeviceManager(DeviceManager const&);
        void operator=(DeviceManager const&);

        // Attributes
        std::string    devInfo;

        std::vector<cl::CommandQueue>     queues;
        std::vector<cl::Platform>      platforms;
        std::vector<cl::Context>        contexts;
        std::vector<unsigned>         ctxOffsets;

        unsigned activeCtxId;
        unsigned activeQId;
};

}
