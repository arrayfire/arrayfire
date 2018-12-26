/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <math.h>
#include <cstdio>

using namespace af;

static const int ITERATIONS  = 50;
static const float PRECISION = 1.0f / ITERATIONS;

int main(int, char**) {
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(800, 800, "2D Plot example: ArrayFire");

        array Y;
        int sign    = 1;
        array X     = seq(-af::Pi, af::Pi, PRECISION);
        array noise = randn(X.dims(0)) / 5.f;

        myWindow.grid(2, 1);

        for (double val = 0; !myWindow.close();) {
            Y = sin(X);

            myWindow(0, 0).plot(X, Y);
            myWindow(1, 0).scatter(X, Y + noise, AF_MARKER_POINT);

            myWindow.show();

            X = X + PRECISION * float(sign);
            val += PRECISION * float(sign);

            if (val > af::Pi) {
                sign = -1;
            } else if (val < -af::Pi) {
                sign = 1;
            }
        }

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
