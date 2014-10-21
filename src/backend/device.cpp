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
    return AF_SUCCESS;
}

af_err af_set_device(const int device)
{
    if(setDevice(device) < 0) {
        std::cout << "Invalid Device ID" << std::endl;
        return AF_ERR_INVALID_ARG;
    }
    return AF_SUCCESS;
}

