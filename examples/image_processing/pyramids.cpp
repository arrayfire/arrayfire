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
#include <cstdlib>

using namespace af;

static const float pyramid_kernel[] = {
    1,  4,  6,  4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1,  4,  6,  4, 1
};

array pyramid(const array& img, const int level, const bool sampling)
{
    array pyr = img.copy();
    array kernel(5, 5, pyramid_kernel);
    kernel = kernel / 256.f;
    if(sampling) {                              //Downsample
        for(int i = 0; i < level; i++) {
            for(int j = 0; j < pyr.dims(2); j++)
                pyr(span, span, j) = convolve(pyr(span, span, j), kernel);
            pyr = pyr(seq(0, pyr.dims(0)-1, 2), seq(0, pyr.dims(1)-1, 2), span);
        }
    } else {                                    // Up sample
        for(int i = 0; i < level; i++) {
            array tmp = constant(0, pyr.dims(0) * 2, pyr.dims(1) * 2, pyr.dims(2));
            tmp(seq(0, 2*pyr.dims(0)-1, 2), seq(0, 2*pyr.dims(1)-1, 2), span) = pyr;
            for(int j = 0; j < pyr.dims(2); j++)
                tmp(span, span, j) = convolve(tmp(span, span, j), kernel * 4.f);
            pyr = tmp;
        }
    }
    return pyr;
}

void pyramids_demo(bool console)
{
    af::Window wnd_rgb("Image Pyramids - RGB Images");
    af::Window wnd_gray("Image Pyramids - Grayscale Images");
    wnd_rgb.setPos(25, 25);
    wnd_gray.setPos(150, 150);

    array img_rgb = loadImage(ASSETS_DIR "/examples/images/atlantis.png", true) / 255.f; // 3 channel RGB       [0-1]
    array img_gray = colorSpace(img_rgb, AF_GRAY, AF_RGB);

    array downc1 = pyramid(img_rgb,  1, true);
    array downc2 = pyramid(img_rgb,  2, true);
    array upc1   = pyramid(img_rgb,  1, false);
    array upc2   = pyramid(img_rgb,  2, false);

    array downg1 = pyramid(img_gray, 1, true);
    array downg2 = pyramid(img_gray, 2, true);
    array upg1   = pyramid(img_gray, 1, false);
    array upg2   = pyramid(img_gray, 2, false);

    while (!wnd_rgb.close() && !wnd_gray.close()) {

        wnd_rgb.grid(2, 3);
        wnd_rgb(0, 0).image(img_rgb, "color image");
        wnd_rgb(1, 0).image(downc1, "downsample 1 level");
        wnd_rgb(0, 1).image(downc2, "downsample 2 levels");
        wnd_rgb(1, 1).image(upc1, "upsample 1 level");
        wnd_rgb(0, 2).image(upc2, "upsample 2 level");
        wnd_rgb.show();


        wnd_gray.grid(2, 3);
        wnd_gray(0, 0).image(img_gray, "grayscale image");
        wnd_gray(1, 0).image(downg1, "downsample 1 level");
        wnd_gray(0, 1).image(downg2, "downsample 2 levels");
        wnd_gray(1, 1).image(upg1, "upsample 1 level");
        wnd_gray(0, 2).image(upg2, "upsample 2 level");
        wnd_gray.show();
    }
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::setDevice(device);
        af::info();
        printf("** ArrayFire Image Pyramids Demo **\n\n");
        pyramids_demo(console);

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
