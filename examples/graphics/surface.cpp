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

static const int M = 30;
static const int N = 2 * M;

int main(int, char**) {
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(800, 800, "3D Surface example: ArrayFire");

        // Creates grid of between [-1 1] with precision of 1 / M
        const array x = iota(dim4(N, 1), dim4(1, N)) / M - 1;
        const array y = iota(dim4(1, N), dim4(N, 1)) / M - 1;

        static float t = 0;
        while (!myWindow.close()) {
            t += 0.07;
            array z = 10 * x * -abs(y) * cos(x * x * (y + t)) +
                      sin(y * (x + t)) - 1.5;
            myWindow.surface(x, y, z);
        }

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
