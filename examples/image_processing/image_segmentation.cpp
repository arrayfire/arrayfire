/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include <arrayfire.h>
#include <af/util.h>
#include <cstdlib>

using namespace af;

bool isColor = true;

array normalize(const array &in)
{
    float mx = max<float>(in.as(f32));
    float mn = min<float>(in.as(f32));
    return (255*(in-mn)/(mx-mn)).as(u8);
}

int main(int argc, char* argv[])
{
    int device = argc > 1 ? atoi(argv[1]) : 0;

    try {
        af::setDevice(device);
        af::info();

        printf("** ArrayFire Image Segmentation Demo **\n");
        af::Window myWindow("Image Segmentation Demo");

        isColor = true;
        array in = loadImage(ASSETS_DIR "/examples/images/atlantis.png", isColor);

        array out = meanShift(in, 15.f, 30.f, 10, isColor);

        while(!myWindow.close()) {

            myWindow.grid(1, 2);

            myWindow(0,0).image(normalize(in) , "Input ");
            myWindow(0,1).image(normalize(out), "MeanShift()");

            myWindow.show();
        }

    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
