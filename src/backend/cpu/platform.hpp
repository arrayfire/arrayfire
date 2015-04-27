/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <string>
#if defined(WITH_GRAPHICS)
#include <fg/window.h>
#endif

namespace cpu {
    std::string getInfo();

    bool isDoubleSupported(int device);

    void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    int getDeviceCount();

    int setDevice(int device);

    int getActiveDeviceId();

    void sync(int device);

#if defined(WITH_GRAPHICS)
    void markDeviceForInterop(const int device, const fg::Window* window);
#endif
}
