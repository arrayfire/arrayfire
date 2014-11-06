/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <af/exception.h>
#include <iostream>

using namespace af;

int main(int argc, char *argv[])
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        printf("\ncreate a 5-by-3 matrix of random floats on the GPU\n");
        array A = randu(5,3, f32);
        af_print(A);

        printf("element-wise arithmetic\n");
        array B = sin(A) + 1.5;
        af_print(B);

        printf("Fourier transform the result\n");
        array C = fft(B);
        af_print(C);

        printf("grab last row\n");
        array c = C.row(end);
        af_print(c);

        printf("zero out every other column\n");
        B(span, seq(0, end, 2)) = 0;
        printf("negate the first three elements of second column\n");
        B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;
        af_print(B);

        printf("create 2-by-3 matrix from host data\n");
        float d[] = { 1, 2, 3, 4, 5, 6 };
        array D(2, 3, d, af::afHost);
        af_print(D);

        printf("copy last column onto first\n");
        D.col(0) = D.col(end);
        af_print(D);

        // Sort A
        printf("sort A and print sorted array and corresponding indices\n");
        array E, F;
        sort(E, F, A);
        af_print(E);
        af_print(F);

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
