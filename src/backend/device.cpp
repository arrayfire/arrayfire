/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/device.h>
#include <backend.hpp>
#include <platform.hpp>
#include <iostream>
#include "err_common.hpp"

using namespace detail;

af_err af_info()
{
    std::cout << getInfo();
    return AF_SUCCESS;
}

af_err af_get_device_count(int *nDevices)
{
    try {
        *nDevices = getDeviceCount();
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device(int *device)
{
    try {
        *device = getActiveDeviceId();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_device(const int device)
{
    try {
        if (setDevice(device) < 0) {
            std::cout << "Invalid Device ID" << std::endl;
            return AF_ERR_INVALID_ARG;
        }
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_sync(const int device)
{
    try {
        int dev = device == -1 ? getActiveDeviceId() : device;
        detail::sync(dev);
    } CATCHALL;
    return AF_SUCCESS;
}
