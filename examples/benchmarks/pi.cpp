/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/*
   monte-carlo estimation of PI

   algorithm:
   - generate random (x,y) samples uniformly
   - count what percent fell inside (top quarter) of unit circle
*/

#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <arrayfire.h>
using namespace af;

// generate millions of random samples
static int samples = 20e6;

/* Self-contained code to run host and device estimates of PI.  Note that
   each is generating its own random values, so the estimates of PI
   will differ. */
static double pi_device()
{
    array x = randu(samples,f32), y = randu(samples,f32);
    return 4.0 * sum<float>(sqrt(x*x + y*y) < 1) / samples;
}

static double pi_host()
{
    int count = 0;
    for (int i = 0; i < samples; ++i) {
        float x = float(rand()) / RAND_MAX;
        float y = float(rand()) / RAND_MAX;
        if (sqrt(x*x + y*y) < 1)
            count++;
    }
    return 4.0 * count / samples;
}



// void wrappers for timeit()
static void device_wrapper() { pi_device(); }
static void host_wrapper() { pi_host(); }


int main(int argc, char ** argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        setDevice(device);
        info();

        printf("device:  %.5f seconds to estimate  pi = %.5f\n", timeit(device_wrapper), pi_device());
        printf("  host:  %.5f seconds to estimate  pi = %.5f\n", timeit(host_wrapper), pi_host());
    } catch (exception& e) {
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
