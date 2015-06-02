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

int main(int argc, char *argv[])
{
    try {
        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        array in = randu(5, 8);
        af_print(in);

        array lin = in.copy();

        printf("Running LU InPlace\n");
        array pivot;
        luInPlace(pivot, lin);
        af_print(lin);
        af_print(pivot);

        printf("Running LU with Upper Lower Factorization\n");
        array lower, upper;
        lu(lower, upper, pivot, in);
        af_print(lower);
        af_print(upper);
        af_print(pivot);

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
    #endif
    return 0;
}
