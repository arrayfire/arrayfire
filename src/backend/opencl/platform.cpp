#define __CL_ENABLE_EXCEPTIONS
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

using std::cout;
using std::endl;
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

DeviceManager& DeviceManager::getInstance() {
    static DeviceManager my_instance;
    return my_instance;
}

string DeviceManager::getInfo() const {
    return devInfo;
}

unsigned DeviceManager::deviceCount() const {
    return queues.size();
}

unsigned DeviceManager::getActiveDeviceId() const {
    return activeQId;
}

const Platform& DeviceManager::getActivePlatform() const {
    return platforms[activeCtxId];
}

const Context& DeviceManager::getActiveContext() const {
    return contexts[activeCtxId];
}

CommandQueue& DeviceManager::getActiveQueue() {
    return queues[activeQId];
}

void DeviceManager::setDevice(size_t device) {
    if (device>=queues.size() || device>=DeviceManager::MAX_DEVICES)
        throw runtime_error("@DeviceManger: invalid device index");
    else {
        activeQId = device;
        for(size_t i=0; i<ctxOffsets.size(); ++i) {
            if (device<ctxOffsets[i]) {
                activeCtxId = i;
                break;
            }
        }
    }
}

DeviceManager::DeviceManager() : devInfo(""), activeCtxId(0), activeQId(0) {
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
            info<<" Device driver "<<dev.getInfo<CL_DRIVER_VERSION>()<<endl;

            queues.emplace_back(context, dev);
        }
        pIter++;

        ctxOffsets.push_back(nDevices);
    }
    devInfo = info.str();
}

CommandQueue& getQueue()
{
    return DeviceManager::getInstance().getActiveQueue();
}

const Context& getContext()
{
    return DeviceManager::getInstance().getActiveContext();
}

unsigned getDeviceCount()
{
    return DeviceManager::getInstance().deviceCount();
}

unsigned getCurrentDeviceId()
{
    return DeviceManager::getInstance().getActiveDeviceId();
}

}

af_err af_info()
{
    af_err ret = AF_SUCCESS;

    try {
        cout<<opencl::DeviceManager::getInstance().getInfo()<<endl;
    }
    CATCHALL;

    return ret;
}

af_err af_get_device_count(int *num_of_devices)
{
    af_err ret = AF_SUCCESS;

    try {
        *num_of_devices = opencl::getDeviceCount();
    }
    CATCHALL;

    return ret;
}

af_err af_set_device(int device)
{
    af_err ret = AF_SUCCESS;

    try {
        opencl::DeviceManager::getInstance().setDevice(device);
    }
    CATCHALL;

    return ret;
}
