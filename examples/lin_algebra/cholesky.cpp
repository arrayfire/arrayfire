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

        int n = 5;
        array t = randu(n, n);
        array in = matmulNT(t, t) + identity(n, n) * n;
        af_print(in);

        printf("Running Cholesky InPlace\n");
        array cin = in.copy();
        af_print(cin);

        printf("Running Cholesky Out of place\n");
        array out_upper;
        array out_lower;

        cholesky(out_upper, in, true);
        cholesky(out_lower, in, false);

        af_print(out_upper);
        af_print(out_lower);

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
