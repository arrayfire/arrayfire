/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

using namespace af;

int main(int argc, char* argv[])
{
    try {
        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        float h_buffer[] = {1, 4, 2, 5, 3, 6 };  // host array
        array in(2, 3, h_buffer);                        // copy host data to device
        print("in", in);

        array s;
        array u;
        array vt;
        printf("Running svd");
        svd(s, u, vt, in);
        print("in", in);
        print("s", s);
        print("u", u);
        print("vt", vt);

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

#ifdef WIN32  // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
#endif
    return 0;
}
