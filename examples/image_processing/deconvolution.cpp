/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <stdio.h>
#include <af/util.h>
#include <cstdlib>

using namespace af;

const unsigned ITERATIONS     = 96;
const float RELAXATION_FACTOR = 0.05f;

array normalize(const array &in) {
    float mx = max<float>(in.as(f32));
    float mn = min<float>(in.as(f32));
    return (in - mn) / (mx - mn);
}

int main(int argc, char *argv[]) {
    int device = argc > 1 ? atoi(argv[1]) : 0;

    try {
        af::setDevice(device);
        af::info();

        printf("** ArrayFire Image Deconvolution Demo **\n");
        af::Window myWindow("Image Deconvolution");

        array in = loadImage(ASSETS_DIR "/examples/images/house.jpg", false);
        array kernel  = gaussianKernel(13, 13, 2.25, 2.25);
        array blurred = convolve(in, kernel);
        array tikhonov =
            inverseDeconv(blurred, kernel, 0.05, AF_INVERSE_DECONV_TIKHONOV);

        array landweber =
            iterativeDeconv(blurred, kernel, ITERATIONS, RELAXATION_FACTOR,
                            AF_ITERATIVE_DECONV_LANDWEBER);

        array richlucy =
            iterativeDeconv(blurred, kernel, ITERATIONS, RELAXATION_FACTOR,
                            AF_ITERATIVE_DECONV_RICHARDSONLUCY);

        while (!myWindow.close()) {
            myWindow.grid(2, 3);

            myWindow(0, 0).image(normalize(in), "Input Image");
            myWindow(1, 0).image(normalize(blurred), "Blurred Image");
            myWindow(0, 1).image(normalize(tikhonov), "Tikhonov");
            myWindow(1, 1).image(normalize(landweber), "Landweber");
            myWindow(0, 2).image(normalize(richlucy), "Richardson-Lucy");

            myWindow.show();
        }

    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
