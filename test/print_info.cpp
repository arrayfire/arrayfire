/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>

using namespace af;

int main(int, const char**) {
    int backend = getAvailableBackends();
    if (backend & AF_BACKEND_OPENCL) {
        setBackend(AF_BACKEND_OPENCL);
    } else if (backend & AF_BACKEND_CUDA) {
        setBackend(AF_BACKEND_CUDA);
    } else if (backend & AF_BACKEND_CPU) {
        setBackend(AF_BACKEND_CPU);
    }

    info();
    return 0;
}
