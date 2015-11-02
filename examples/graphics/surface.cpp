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

static const int ITERATIONS = 30;
static const float PRECISION = 1.0f/ITERATIONS;

int main(int argc, char *argv[])
{
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(800, 800, "3D Surface example: ArrayFire");

        array X = seq(-1, 1, PRECISION);
        array Y = seq(-1, 1, PRECISION);
        array Z = randn(X.dims(0), Y.dims(0));

        static float t=0;
        while(!myWindow.close()) {
            t+=0.07;
            //Z = sin(tile(X,1, Y.dims(0))*t + t) + cos(transpose(tile(Y, 1, X.dims(0)))*t + t);
            array x = tile(X,1, Y.dims(0));
            array y = transpose(tile(Y, 1, X.dims(0)));
            Z = 10*x*-abs(y) * cos(x*x*(y+t))+sin(y*(x+t))-1.5;

            myWindow.surface(X, Y, Z, NULL);
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

