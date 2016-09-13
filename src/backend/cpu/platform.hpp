/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <string>

namespace cpu {
    class queue;

    int getBackend();

    std::string getDeviceInfo();

    bool isDoubleSupported(int device);

    void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    int getDeviceCount();

    int setDevice(int device);

    int getActiveDeviceId();

    size_t getDeviceMemorySize(int device);

    size_t getHostMemorySize();

    void sync(int device);

    queue& getQueue(int idx = 0);

    unsigned getMaxJitSize();

    bool& evalFlag();
}
