/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cstdio>
#include <arrayfire.h>
#include <cstdlib>

using namespace af;

static void fast_demo(bool console)
{
    // Load image
    array img_color;
    if (console)
        img_color = loadImage(ASSETS_DIR "/examples/images/square.png", true);
    else
        img_color = loadImage(ASSETS_DIR "/examples/images/lena.ppm", true);
    // Convert the image from RGB to gray-scale
    array img = colorSpace(img_color, AF_GRAY, AF_RGB);
    // For visualization in ArrayFire, color images must be in the [0.0f-1.0f] interval
    img_color /= 255.f;

    features feat = fast(img, 20.0f, 9, true, 0.05);

    float* h_x = feat.getX().host<float>();
    float* h_y = feat.getY().host<float>();

    // Draw draw_len x draw_len crosshairs where the corners are
    const int draw_len = 3;
    for (size_t f = 0; f < feat.getNumFeatures(); f++) {
        int x = h_x[f];
        int y = h_y[f];
        img_color(y, seq(x-draw_len, x+draw_len), 0) = 0.f;
        img_color(y, seq(x-draw_len, x+draw_len), 1) = 1.f;
        img_color(y, seq(x-draw_len, x+draw_len), 2) = 0.f;

        // Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
        // Set only the first channel to 1 (green lines)
        img_color(seq(y-draw_len, y+draw_len), x, 0) = 0.f;
        img_color(seq(y-draw_len, y+draw_len), x, 1) = 1.f;
        img_color(seq(y-draw_len, y+draw_len), x, 2) = 0.f;
    }

    printf("Features found: %lu\n", feat.getNumFeatures());

    if (!console) {
        af::Window wnd("FAST Feature Detector");

        // Previews color image with green crosshairs
        while(!wnd.close())
            wnd.image(img_color);
    } else {
        af_print(feat.getX());
        af_print(feat.getY());
    }
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::setDevice(device);
        af::info();
        std::cout << "** ArrayFire FAST Feature Detector Demo **" << std::endl << std::endl;
        fast_demo(console);

    } catch (af::exception& ae) {
        std::cerr << ae.what() << std::endl;
        throw;
    }

    return 0;
}
