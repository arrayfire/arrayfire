/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <CL/cl.h>
#include <af/defines.h>

namespace afcl
{
    AFAPI cl_context getContext();
    AFAPI cl_command_queue getQueue();
    AFAPI cl_device_id getDeviceId();
}
