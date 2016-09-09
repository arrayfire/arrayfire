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

const float MINIMUM = 1.0f;
const float MAXIMUM = 20.f;
const float STEP    = 2.0f;
const float NELEMS  = (MAXIMUM-MINIMUM+1)/STEP;

int main(int argc, char *argv[])
{
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(1024, 1024, "2D Vector Field example: ArrayFire");

        float h_divPoints[] = {5, 5, 15, 15,
                               5, 15, 5, 15};
        array divPoints(4, 2, h_divPoints);

        //array points = join(1, flat(range(dim4(10, 10)) * 2 + 1), flat(range(dim4(10, 10), 1) * 2 + 1));
        array points = join(1, flat(range(dim4(10, 10), 1) * 2 + 1), flat(range(dim4(10, 10)) * 2 + 1));
        array directions = sin(2 * Pi * points / 10.0f);

        myWindow.setAxesLimits(points.col(0), points.col(1));
        myWindow.setAxesTitles();

        while(!myWindow.close()) {
            myWindow.scatter(divPoints, AF_MARKER_CIRCLE);
            myWindow.vectorField(points, directions);
            myWindow.show();
        }

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}

