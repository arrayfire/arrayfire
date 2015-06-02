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
    array B = matmul(A, A);  // matrix multiply
    B.eval();                // ensure evaluated
}

int main(int argc, char ** argv)
{
    double peak = 0;
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        setDevice(device);
        info();

        printf("Benchmark N-by-N matrix multiply\n");
        for (int n = 128; n <= 2048; n += 128) {

            printf("%4d x %4d: ", n, n);
            A = constant(1,n,n);
            double time = timeit(fn); // time in seconds
            double gflops = 2.0 * powf(n,3) / (time * 1e9);
            if (gflops > peak)
                peak = gflops;

            printf(" %4.0f Gflops\n", gflops);
            fflush(stdout);
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    if (argc == 2 && argv[1][0] == '-')
        printf(" ### peak %g GFLOPS\n", peak);

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
    #endif
    return 0;
}
