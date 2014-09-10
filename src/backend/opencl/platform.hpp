#pragma once

#include <cl.hpp>
#include <vector>
#include <string>

namespace opencl
{

class DeviceManager
{
    public:
        static const unsigned MAX_DEVICES = 16;

        static DeviceManager& getInstance();

        std::string getInfo() const;

        unsigned deviceCount() const;

        unsigned getActiveDeviceId() const;

        const cl::Platform& getActivePlatform() const;

        const cl::Context& getActiveContext() const;

        cl::CommandQueue& getActiveQueue();

        void setDevice(size_t device);

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

cl::CommandQueue& getQueue();

const cl::Context& getContext();

unsigned getDeviceCount();

unsigned getCurrentDeviceId();

}
