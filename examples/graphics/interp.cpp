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
#include <iostream>

using namespace af;

int main(int argc, char *argv[])
{
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(1024, 512, "2D Plot example: ArrayFire");

        array X = af::randu(16, 16);
        array X2 = seq(0, 15, 0.1);
        array Y2 = seq(0, 15, 0.1);
        X2 = tile(X2, 1, X2.dims(0));
        Y2 = tile(Y2.T(), Y2.dims(0));
        af::array I1 = approx2(X, X2, Y2, AF_INTERP_NEAREST, 0.0f);
        af::array I2 = approx2(X, X2, Y2, AF_INTERP_LINEAR, 0.0f);
        af::array I3 = approx2(X, X2, Y2, AF_INTERP_CUBIC, 0.0f);
        myWindow.grid(1, 4);
        while (!myWindow.close()) {
            myWindow(0,0).image(X);
            myWindow(0,1).image(I1);
            myWindow(0,2).image(I2);
            myWindow(0,3).image(I3);
            myWindow.show();
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
