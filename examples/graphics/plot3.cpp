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

static const int ITERATIONS  = 200;
static const float PRECISION = 1.0f / ITERATIONS;

int main(int, char**) {
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(800, 800, "3D Line Plot example: ArrayFire");

        static float t = 0.1;
        array Z        = seq(0.1f, 10.f, PRECISION);

        do {
            array Y = sin((Z * t) + t) / Z;
            array X = cos((Z * t) + t) / Z;
            X       = max(min(X, 1.0), -1.0);
            Y       = max(min(Y, 1.0), -1.0);

            // Pts can be passed in as a matrix in the form n x 3, 3 x n
            // or in the flattened xyz-triplet array with size 3n x 1
            myWindow.plot(X, Y, Z);

            t += 0.01;
        } while (!myWindow.close());

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
