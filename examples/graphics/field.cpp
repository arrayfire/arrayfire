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

const static float MINIMUM = -3.0f;
const static float MAXIMUM = 3.0f;
const static float STEP    = 0.18f;

int main(int, char**) {
    try {
        af::info();
        af::Window myWindow(1024, 1024, "2D Vector Field example: ArrayFire");

        myWindow.grid(2, 2);

        array dataRange = seq(MINIMUM, MAXIMUM, STEP);

        array x = tile(dataRange, 1, dataRange.dims(0));
        array y = tile(dataRange.T(), dataRange.dims(0), 1);
        x.eval();
        y.eval();

        float scale = 2.0f;
        do {
            array points = join(1, flat(x), flat(y));

            array saddle = join(1, flat(x), -1.0f * flat(y));

            array bvals = sin(scale * (x * x + y * y));
            array hbowl = join(1, constant(1., x.elements()), flat(bvals));
            hbowl.eval();

            // 2D points
            myWindow(0, 0).vectorField(points, saddle, "Saddle point");
            myWindow(0, 1).vectorField(
                points, hbowl, "hilly bowl (in a loop with varying amplitude)");

            // 2D coordinates
            myWindow(1, 0).vectorField(2.0 * flat(x), flat(y), flat(x),
                                       -flat(y), "Saddle point");
            myWindow(1, 1).vectorField(
                2.0 * flat(x), flat(y), constant(1., x.elements()), flat(bvals),
                "hilly bowl (in a loop with varying amplitude)");

            myWindow.show();

            scale -= 0.0010f;
            if (scale < -0.01f) { scale = 2.0f; }
        } while (!myWindow.close());

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
