#include <cl.hpp>
#include <platform.hpp>
#include <af/opencl.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <errorcodes.hpp>
#include <../helper.hpp>

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

DeviceManager& DeviceManager::getInstance()
{
    static DeviceManager my_instance;
    return my_instance;
}

DeviceManager::DeviceManager()
    : devInfo(""), activeCtxId(0), activeQId(0)
{
    ostringstream info;

    vector<Platform> platforms;
    Platform::get(&platforms);
    vector<string> pnames;

    for (auto platform: platforms) {
        string pstr;
        platform.getInfo(CL_PLATFORM_NAME, &pstr);
        pnames.push_back(pstr);

        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};
        contexts.emplace_back(CL_DEVICE_TYPE_ALL, cps);
    }

    unsigned nDevices = 0;
    vector<string>::iterator pIter = pnames.begin();
    for (auto context: contexts) {
        vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        for (auto dev: devices) {
            string dstr;
            dev.getInfo(CL_DEVICE_NAME, &dstr);

            info<< nDevices++ <<". "<<*pIter<<" "<<dstr<<" ";
            info<<dev.getInfo<CL_DEVICE_VERSION>();
            info<<" Device driver "<<dev.getInfo<CL_DRIVER_VERSION>()<<std::endl;

            queues.emplace_back(context, dev);
        }
        pIter++;

        ctxOffsets.push_back(nDevices);
    }
    devInfo = info.str();
}

std::string getInfo()
{
    return DeviceManager::getInstance().devInfo;
}

unsigned deviceCount()
{
    return DeviceManager::getInstance().queues.size();
}

unsigned getActiveDeviceId()
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

void setDevice(size_t device)
{
    DeviceManager& devMngr = DeviceManager::getInstance();

    if (device>=devMngr.queues.size() ||
            device>=DeviceManager::MAX_DEVICES) {
        throw runtime_error("@setDevice: invalid device index");
    }
    else {
        devMngr.activeQId = device;
        for(size_t i=0; i<devMngr.ctxOffsets.size(); ++i) {
            if (device<devMngr.ctxOffsets[i]) {
                devMngr.activeCtxId = i;
                break;
            }
        }
    }
}

}

af_err af_info()
{
    af_err ret = AF_SUCCESS;

    try {
        std::cout<<opencl::getInfo()<<std::endl;
    }
    CATCHALL;

    return ret;
}

af_err af_get_device_count(int *num_of_devices)
{
    af_err ret = AF_SUCCESS;

    try {
        *num_of_devices = opencl::deviceCount();
    }
    CATCHALL;

    return ret;
}

af_err af_set_device(int device)
{
    af_err ret = AF_SUCCESS;

    try {
        opencl::setDevice(device);
    }
    CATCHALL;

    return ret;
}
