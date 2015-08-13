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

array morphopen(const array& img, const array& mask)
{
    return dilate(erode(img, mask), mask);
}

array morphclose(const array& img, const array& mask)
{
    return erode(dilate(img, mask), mask);
}

array morphgrad(const array& img, const array& mask)
{
    return (dilate(img, mask) - erode(img, mask));
}

array tophat(const array& img, const array& mask)
{
    return (img - morphopen(img, mask));
}

array bottomhat(const array& img, const array& mask)
{
    return (morphclose(img, mask) - img);
}

array border(const array& img, const int left, const int right,
        const int top, const int bottom,
        const float value = 0.0)
{
    if((int)img.dims(0) < (top + bottom))
        std::cerr << "input does not have enough rows" << std::endl;
    if((int)img.dims(1) < (left + right))
        std::cerr << "input does not have enough columns" << std::endl;

    dim4 imgDims = img.dims();
    array ret = constant(value, imgDims);
    ret(seq(top, imgDims[0]-bottom), seq(left, imgDims[1]-right), span, span) =
        img(seq(top, imgDims[0]-bottom), seq(left, imgDims[1]-right), span, span);

    return ret;
}

array border(const array& img, const int w, const int h,
        const float value = 0.0)
{
    return border(img, w, w, h, h, value);
}

array border(const array& img, const int size, const float value = 0.0)
{
    return border(img, size, size, size, size, value);
}

array blur(const array& img, const array mask = gaussianKernel(3,3))
{
    array blurred = array(img.dims(), img.type());
    for(int i = 0; i < (int)blurred.dims(2); i++)
        blurred(span, span, i) = convolve(img(span, span, i), mask);
    return blurred;
}

// Demonstrates various image morphing manipulations.
static void morphing_demo(bool console)
{
    af::Window wnd(1280, 720, "Morphological Operations");
    // load images
    array img_rgb = loadImage(ASSETS_DIR "/examples/images/man.jpg", true) / 255.f; // 3 channel RGB       [0-1]

    array mask = constant(1, 5, 5);

    array er = erode(img_rgb, mask);
    array di = dilate(img_rgb, mask);
    array op = morphopen(img_rgb, mask);
    array cl = morphclose(img_rgb, mask);
    array gr = morphgrad(img_rgb, mask);
    array th = tophat(img_rgb, mask);
    array bh = bottomhat(img_rgb, mask);
    array bl = blur(img_rgb, gaussianKernel(5,5));
    array bp = border(img_rgb, 20, 30, 40, 50, 0.5);
    array bo = border(img_rgb, 20);

    while (!wnd.close()) {
        wnd.grid(3, 4);

        wnd(0, 0).image(img_rgb, "Input"          );
        wnd(1, 0).image(er     , "Erosion"        );
        wnd(2, 0).image(di     , "Dilation"       );

        wnd(0, 1).image(op     , "Opening"        );
        wnd(1, 1).image(cl     , "Closing"        );
        wnd(2, 1).image(gr     , "Gradient"       );

        wnd(0, 2).image(th     , "TopHat"         );
        wnd(1, 2).image(bh     , "BottomHat"      );
        wnd(2, 2).image(bl     , "Blur"           );

        wnd(0, 3).image(bp     , "Border to Gray" );
        wnd(1, 3).image(bo     , "Border to black");

        wnd.show();
    }
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::info();
        af::setDevice(device);
        printf("** ArrayFire Image Morphing Demo **\n\n");
        morphing_demo(console);

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
