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

using namespace af;

void prewitt(array &mag, array &dir, const array &in)
{
    static float h1[] = { 1, 1, 1};
    static float h2[] = {-1, 0, 1};
    static array colf(3, 1, h1);
    static array rowf(3, 1, h2);

    // Find the gradients
    array Gy = convolve(rowf, colf, in);
    array Gx = convolve(colf, rowf, in);

    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}

void sobelFilter(array &mag, array &dir, const array &in)
{
    array Gx, Gy;
    sobel(Gx, Gy, in, 3);
    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}

array normalize(const array &in)
{
    float mx = max<float>(in);
    float mn = min<float>(in);
    return (in-mn)/(mx-mn);
}

array edge(const array &in, int method = 0)
{
    int w = 5;
    if (in.dims(0) <  512) w = 3;
    if (in.dims(0) > 2048) w = 7;

    int h = 5;
    if (in.dims(0) <  512) h = 3;
    if (in.dims(0) > 2048) h = 7;

    array ker = gaussiankernel(w, h);
    array smooth = convolve(in, ker);
    array mag, dir;

    switch(method) {
        case  1: prewitt(mag, dir, smooth); break;
        case  2: sobelFilter(mag, dir, smooth);   break;
        default: throw af::exception("Unsupported type");
    }

    return normalize(mag);
}

void edge(bool console)
{
    array in = loadimage(ASSETS_DIR "/examples/images/lena.ppm", false);

    array prewitt = edge(in, 1);
    array sobelFilter   = edge(in, 2);
    array hst = histogram(in, 256, 0, 255);

    while(!console) {
        // colormap, grayscale
        setupGrid(2, 2);

        bindCell(1, 1); image(in/255);
        bindCell(1, 2); hist(hst, 0, 255);
        bindCell(2, 1); image(prewitt);
        bindCell(2, 2); image(sobelFilter);

        showGrid();
        //FIXME add timeout
    }
}

int main(int argc, char* argv[])
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::deviceset(device);
        af::info();
        af::initGraphics(device);

        printf("** ArrayFire Edge Detection Demo **\n");
        edge(console);

    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
