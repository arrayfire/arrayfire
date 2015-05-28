/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <stdio.h>
#include <math.h>
#include <cstdlib>

using namespace af;

// create a small wrapper to benchmark
static array A; // populated before each timing
static void fn()
{
    array B = fft2(A);  // matrix multiply
    B.eval();           // ensure evaluated
}

int main(int argc, char ** argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        setDevice(device);
        info();

        printf("Benchmark N-by-N 2D fft\n");
        for (int M = 7; M <= 12; M++) {
            int N = (1 << M);

            printf("%4d x %4d: ", N, N);
            A = randu(N,N);
            double time = timeit(fn); // time in seconds
            double gflops = 10.0 * N * N * M / (time * 1e9);

            printf(" %4.0f Gflops\n", gflops);
            fflush(stdout);
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
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
