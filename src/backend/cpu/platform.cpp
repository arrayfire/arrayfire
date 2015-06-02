/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/version.h>
#include <platform.hpp>
#include <sstream>

namespace cpu
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

std::string getInfo()
{
    std::ostringstream info;
    info << "ArrayFire v" << AF_VERSION
         << " (CPU, " << get_system() << ", build " << AF_REVISION << ")" << std::endl;
    return info.str();
}

bool isDoubleSupported(int device)
{
    return true;
}

void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    static bool flag;
    if(!flag) {
        printf("WARNING: af_devprop not supported for CPU\n");
        flag = 1;
    }
}

int getDeviceCount()
{
    return 1;
}


int setDevice(int device)
{
    static bool flag;
    if(!flag) {
        printf("WARNING: af_set_device not supported for CPU\n");
        flag = 1;
    }
    return 1;
}

int getActiveDeviceId()
{
    return 0;
}

void sync(int device)
{
    // Nothing here
}

}
