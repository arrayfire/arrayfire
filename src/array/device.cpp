/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/device.h>
#include <af/compatible.h>
#include "error.hpp"

namespace af
{
    void info()
    {
        AF_THROW(af_info());
    }

    int getDeviceCount()
    {
        int devices = -1;
        AF_THROW(af_get_device_count(&devices));
        return devices;
    }

    int devicecount() { return getDeviceCount(); }

    void setDevice(const int device)
    {
        AF_THROW(af_set_device(device));
    }

    void deviceset(const int device) { setDevice(device); }

    int getDevice()
    {
        int device = 0;
        AF_THROW(af_get_device(&device));
        return device;
    }

    int deviceget() { return getDevice(); }

    void sync(int device)
    {
        AF_THROW(af_sync(device));
    }
}
