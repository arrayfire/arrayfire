/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/version.h>
#include <cl.hpp>
#include <platform.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <errorcodes.hpp>
#include <err_opencl.hpp>

using std::string;
using std::vector;
using std::ostringstream;
using std::runtime_error;

using cl::Platform;
using cl::Context;
using cl::CommandQueue;
using cl::Device;

namespace opencl
{
static const char *get_system(void)
{
    return
#if defined(ARCH_32)
    "32-bit "
#elif defined(ARCH_64)
    "64-bit "
#endif

#if defined(OS_LNX)
    "Linux";
#elif defined(OS_WIN)
    "Windows";
#elif defined(OS_MAC)
    "Mac OSX";
#endif
}

DeviceManager& DeviceManager::getInstance()
{
    static DeviceManager my_instance;
    return my_instance;
}

DeviceManager::DeviceManager()
    : activeCtxId(0), activeQId(0)
{
    Platform::get(&platforms);

    for (auto platform: platforms) {
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};
        contexts.emplace_back(CL_DEVICE_TYPE_ALL, cps);
    }

    unsigned nDevices = 0;
    for (auto context: contexts) {
        vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        for(auto dev : devices) {
            nDevices++;
            queues.emplace_back(context, dev);
        }

        ctxOffsets.push_back(nDevices);
    }
}

std::string getInfo()
{
    ostringstream info;
    info << "ArrayFire v" << AF_VERSION << AF_VERSION_MINOR
         << " (OpenCL, " << get_system() << ", build " << REVISION << ")" << std::endl;

    vector<string> pnames;
    for (auto platform: DeviceManager::getInstance().platforms) {
        string pstr;
        platform.getInfo(CL_PLATFORM_NAME, &pstr);
        pnames.push_back(pstr);
    }

    unsigned nDevices = 0;
    vector<string>::iterator pIter = pnames.begin();
    for (auto context: DeviceManager::getInstance().contexts) {
        vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        for(unsigned i = 0; i < devices.size(); i++) {
            bool show_braces = ((unsigned)getActiveDeviceId() == nDevices);
            string dstr;
            devices[i].getInfo(CL_DEVICE_NAME, &dstr);

            string id = (show_braces ? string("[") : "-") + std::to_string(nDevices++) +
                        (show_braces ? string("]") : "-");
            info << id << " " << *pIter << " " << dstr << " ";
            info << devices[i].getInfo<CL_DEVICE_VERSION>();
            info << " Device driver " << devices[i].getInfo<CL_DRIVER_VERSION>() <<std::endl;
        }
        pIter++;
    }
    return info.str();
}

int getDeviceCount()
{
    return DeviceManager::getInstance().queues.size();
}

int getActiveDeviceId()
{
    return DeviceManager::getInstance().activeQId;
}


const Context& getContext()
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    return devMngr.contexts[devMngr.activeCtxId];
}

CommandQueue& getQueue()
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    return devMngr.queues[devMngr.activeQId];
}

int setDevice(int device)
{
    DeviceManager& devMngr = DeviceManager::getInstance();

    if (device>= (int)devMngr.queues.size() ||
            device>= (int)DeviceManager::MAX_DEVICES) {
        //throw runtime_error("@setDevice: invalid device index");
        return -1;
    }
    else {
        int old = devMngr.activeQId;
        devMngr.activeQId = device;
        for(int i=0; i< (int)devMngr.ctxOffsets.size(); ++i) {
            if (device< (int)devMngr.ctxOffsets[i]) {
                devMngr.activeCtxId = i;
                break;
            }
        }
        return old;
    }
}

void sync(int device)
{
    try {
        int currDevice = getActiveDeviceId();
        setDevice(device);
        getQueue().finish();
        setDevice(currDevice);
    } catch (cl::Error ex) {
        CL_TO_AF_ERROR(ex);
    }
}

}
