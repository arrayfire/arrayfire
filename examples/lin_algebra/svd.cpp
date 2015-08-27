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

        array u;
        array s_vec;
        array vt;
        svd(u, s_vec, vt, in);

        array s_mat = diag(s_vec, 0, false);
        array in_recon = matmul(u, s_mat, vt(seq(2), span));

        af_print(in);
        af_print(s_vec);
        af_print(u);
        af_print(s_mat);
        af_print(vt);
        af_print(in_recon);

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
