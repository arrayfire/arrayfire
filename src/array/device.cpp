/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/device.h>
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

    void setDevice(const int device)
    {
        AF_THROW(af_set_device(device));
    }
}
