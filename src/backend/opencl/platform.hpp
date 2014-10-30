/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cl.hpp>
#include <vector>
#include <string>

namespace opencl
{

std::string getInfo();

int getDeviceCount();

unsigned getActiveDeviceId();

const cl::Context& getContext();

cl::CommandQueue& getQueue();

int setDevice(int device);

class DeviceManager
{
    friend std::string getInfo();

    friend int getDeviceCount();

    friend unsigned getActiveDeviceId();

    friend const cl::Context& getContext();

    friend cl::CommandQueue& getQueue();

    friend int setDevice(int device);

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
        std::vector<cl::CommandQueue>     queues;
        std::vector<cl::Platform>      platforms;
        std::vector<cl::Context>        contexts;
        std::vector<unsigned>         ctxOffsets;

        unsigned activeCtxId;
        unsigned activeQId;
};

}
