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

static const int ITERATIONS = 50;
static const float PRECISION = 1.0f/ITERATIONS;

int main(int argc, char *argv[])
{
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(1024, 512, "2D Plot example: ArrayFire");

        array Y;
        int sign = 1;
        array X = seq(-af::Pi, af::Pi, PRECISION);
        array noise = randn(X.dims(0))/5.f;

        myWindow.grid(1, 2);
        for (double val=-af::Pi; !myWindow.close(); ) {

            Y = sin(X);

            myWindow(0,0).plot(X, Y);
            myWindow(0,1).scatter(X, Y + noise, AF_MARKER_POINT);

            myWindow.show();

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
    return 0;
}
