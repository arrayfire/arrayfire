/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
    AFAPI void info();

    AFAPI int getDeviceCount();

    AFAPI void setDevice(const int device);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_info();

    AFAPI af_err af_get_device_count(int *num_of_devices);

    AFAPI af_err af_set_device(const int device);

#ifdef __cplusplus
}
#endif
