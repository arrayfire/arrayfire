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
#include <math.h>

using namespace af;

static const int ITERATIONS = 100;
static const float PRECISION = 1.0f/ITERATIONS;

int main(int argc, char *argv[])
{
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(512, 512, "2D Plot example: ArrayFire");

        array Y;
        int sign = 1;
        array X = seq(-af::Pi, af::Pi, PRECISION);

        for (double val=-af::Pi; !myWindow.close(); ) {

            Y = sin(X);

            myWindow.plot(X, Y);

            X = X + PRECISION * float(sign);
            val += PRECISION * float(sign);

            if (val>af::Pi) {
                sign = -1;
            } else if (val<-af::Pi) {
                sign = 1;
            }
        }

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

