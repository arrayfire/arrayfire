#include <af/device.h>
#include <backend.hpp>
#include <platform.hpp>
#include <iostream>

using namespace detail;

af_err af_info()
{
    std::cout << getInfo();
    return AF_SUCCESS;
}

af_err af_get_device_count(int *nDevices)
{
    *nDevices = getDeviceCount();
    if(nDevices <= 0) {
        return AF_ERR_RUNTIME;
    } else {
        return AF_SUCCESS;
    }
}

af_err af_set_device(const int device)
{
    if(setDevice(device) < 0) {
        return AF_ERR_RUNTIME;
    } else {
        return AF_SUCCESS;
    }
}

