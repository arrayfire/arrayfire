/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include <iostream>
#include <arrayfire.h>
#include <math.h>

#define WIDTH 600 // Width of image
#define HEIGHT 800 // Width of image

using namespace af;

array complex_grid(int width, int height, float zoom, float center[2])
{

    // Generate sequences of length width, height
    array x = (seq(double(height)) - double(height) / 2.0);
    array y = (seq(double(width )) - double(width)  / 2.0);

    // Tile the sequences to generate grid of image size
    array X = tile(x.T(), y.elements(), 1) / zoom + center[0];
    array Y = tile(y    , 1, x.elements()) / zoom + center[1];

    // Return the locations as a complex grid
    return complex(X, Y);
}

array mandelbrot(const array &in, int iter, float maxval)
{
    array C = in;
    array Z = C;
    array mag = constant(0, C.dims());

    for (int ii = 1; ii < iter; ii++) {

        // Do the calculation
        Z = Z * Z + C;

        // Get indices where abs(Z) crosses maxval
        array cond = (abs(Z) > maxval).as(f32);
        mag = af::max(mag, cond * ii);

        // If abs(Z) cross maxval, turn off those locations
        C = C * (1 - cond);
        Z = Z * (1 - cond);
    }

    // Normalize
    return mag / maxval;
}

int main(int argc, char **argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    int iter = argc > 2 ? atoi(argv[2]) : 100;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    try {
        af::setDevice(device);
        info();
        printf("** ArrayFire Fractals Demo **\n");
        af::Window wnd("Fractal Demo");

        float center[] = {-0.5, 0};
        // Keep zomming out for each frame
        for (int zoom = 1000; zoom > 10; zoom -= 5) {
            // Generate the grid at the current zoom factor
            array c = complex_grid(WIDTH, HEIGHT, zoom, center);

            // Generate the mandelbrot image
            array mag = mandelbrot(c, iter, 1000);
            if(!console) {
                wnd.image(mag);
            }
        }

    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
