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

        printf("Running QR InPlace\n");
        array in = randu(5, 8);
        af_print(in);

        array qin = in.copy();

        array tau;
        qrInPlace(tau, qin);

        af_print(qin);
        af_print(tau);

        printf("Running QR with Q and R factorization\n");
        array q, r;
        qr(q, r, tau, in);

        af_print(q);
        af_print(r);
        af_print(tau);

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
