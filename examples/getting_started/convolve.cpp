/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include <cstdlib>
#include <arrayfire.h>
using namespace af;

// use static variables at file scope so timeit() wrapper functions
// can reference image/kernels

// image to convolve
static array img;

// 5x5 derivative with separable kernels
static float h_dx[] = {1.f / 12, -8.f / 12, 0, 8.f / 12, -1.f / 12}; // five point stencil
static float h_spread[] = {1.f / 5, 1.f / 5, 1.f / 5, 1.f / 5, 1.f / 5};
static array dx, spread, kernel; // device kernels

static array full_out, dsep_out, hsep_out; // save output for value checks
// wrapper functions for timeit() below
static void full() { full_out = convolve2(img, kernel);}
static void dsep() { dsep_out = convolve(dx, spread, img); }

static bool fail(array &left, array &right)
{
    return (max<float>(abs(left - right)) > 1e-6);
}

int main(int argc, char **argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        // setup image and device copies of kernels
        img = randu(640, 480);
        dx = array(5, 1, h_dx); // 5x1 kernel
        spread = array(1, 5, h_spread); // 1x5 kernel
        kernel = matmul(dx, spread); // 5x5 kernel

        printf("full 2D convolution:         %.5f seconds\n", timeit(full));
        printf("separable, device pointers:  %.5f seconds\n", timeit(dsep));

        // ensure values are all the same across versions
        if (fail(full_out, dsep_out)) { throw af::exception("full != dsep"); }
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
