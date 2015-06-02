/*******************************************************
* Copyright (c) 2015, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <arrayfire.h>

using namespace af;

array clamp(const array &in, float min = 0.0f, float max = 255.0f)
{
    return ((in<min)*0.0f + (in>max)*255.0f + (in >= min && in <= max)*in);
}

/**
* randomization - controls % of total number of pixels in the image
* that will be effected by random noise
* repeat - # of times the process is carried out on the previous steps output
*/
array hurl(const array &in, int randomization, int repeat)
{
    int w = in.dims(0);
    int h = in.dims(1);
    float f = randomization / 100.0f;
    int dim = (int)(f*w*h);
    array ret_val = in.copy();
    array temp = moddims(ret_val, w*h, 3);
    for (int i = 0; i<repeat; ++i) {
        array idxs = (w*h)  * randu(dim);
        array rndR = 255.0f * randu(dim);
        array rndG = 255.0f * randu(dim);
        array rndB = 255.0f * randu(dim);
        temp(idxs, 0) = rndR;
        temp(idxs, 1) = rndG;
        temp(idxs, 2) = rndB;
    }
    ret_val = moddims(temp, in.dims());
    return ret_val;
}

/**
* Retrieve a new image of same dimensions of the original image
* where each original image's pixel is replaced by randomly picked
* neighbor in the provided local neighborhood window
*/
array getRandomNeighbor(const array &in, int windW, int windH)
{
    array rnd = 2.0f*randu(in.dims(0), in.dims(1)) - 1.0f;
    array sx = seq(in.dims(0));
    array sy = seq(in.dims(1));
    array vx = tile(sx, 1, in.dims(1)) + floor(rnd*windW);
    array vy = tile(sy.T(), in.dims(0), 1) + floor(rnd*windH);
    array vxx = clamp(vx, 0, in.dims(0));
    array vyy = clamp(vy, 0, in.dims(1));
    array in2 = moddims(in, vx.elements(), 3);
    return moddims(in2(vyy*in.dims(0) + vxx, span), in.dims());
}

/**
* randomly pick neighbor from given window size and replace the
* current pixel with the randomly chosen color.
* No new colors are introduced, unlike hurl.
*/
array spread(const array &in, int window_width, int window_height)
{
    return getRandomNeighbor(in, window_width, window_height);
}

/**
* randomization - controls % of total number of pixels in the image
* that will be effected by random noise
* repeat - # of times the process is carried out on the previous steps output
*/
array pick(const array &in, int randomization, int repeat)
{
    int w = in.dims(0);
    int h = in.dims(1);
    float f = randomization / 100.0f;
    int dim = (int)(f*w*h);
    array ret_val = in.copy();
    for (int i = 0; i<repeat; ++i) {
        array idxs = (w*h)  * randu(dim);
        array rnd = getRandomNeighbor(ret_val, 1, 1);
        array temp_src = moddims(rnd, w*h, 3);
        array temp_dst = moddims(ret_val, w*h, 3);
        temp_dst(idxs, span) = temp_src(idxs, span);
        ret_val = moddims(temp_dst, in.dims());
    }
    return ret_val;
}

void prewitt(array &mag, array &dir, const array &in)
{
    static float h1[] = { 1, 1, 1 };
    static float h2[] = { -1, 0, 1 };
    static array h1d(3, h1);
    static array h2d(3, h2);

    // Find the gradients
    array Gy = af::convolve(h2d, h1d, in) / 6;
    array Gx = af::convolve(h1d, h2d, in) / 6;

    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}

void sobelFilter(array &mag, array &dir, const array &in)
{
    // Find the gradients
    array Gy, Gx;
    af::sobel(Gx, Gy, in);

    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}

void normalizeImage(array &in)
{
    float min = af::min<float>(in);
    float max = af::max<float>(in);
    in = 255.0f*((in - min) / (max - min));
}

array dog(const array &in, int window_radius1, int window_radius2)
{
    array ret_val;
    int w1 = 2 * window_radius1 + 1;
    int w2 = 2 * window_radius2 + 1;
    array g1 = gaussianKernel(w1, w1);
    array g2 = gaussianKernel(w2, w2);
    ret_val = (convolve(in, g1) - convolve(in, g2));
    normalizeImage(ret_val);
    return ret_val;
}

array medianfilter(const array &in, int window_width, int window_height)
{
    array ret_val(in.dims());
    ret_val(span, span, 0) = medfilt(in(span, span, 0), window_width, window_height);
    ret_val(span, span, 1) = medfilt(in(span, span, 1), window_width, window_height);
    ret_val(span, span, 2) = medfilt(in(span, span, 2), window_width, window_height);
    return ret_val;
}

array gaussianblur(const array &in, int window_width, int window_height, int sigma)
{
    array g = gaussianKernel(window_width, window_height, sigma, sigma);
    return convolve(in, g);
}

/**
* azimuth range is [0-360]
* elevation range is [0-180]
* depth range is [1-100]
* Note: this function has been tailored after
* the emboss implementation in GIMP editor
**/
array emboss(const array &input, float azimuth, float elevation, float depth)
{
    if (depth<1 || depth>100) {
        printf("Depth should be in the range of 1-100");
        return input;
    }
    static float x[3] = { -1, 0, 1 };
    static array hg(3, x);
    static array vg = hg.T();

    array in = input;
    if (in.dims(2)>1)
        in = colorSpace(input, AF_GRAY, AF_RGB);
    else
        in = input;

    // convert angles to radians
    float phi = elevation*af::Pi / 180.0f;
    float theta = azimuth*af::Pi / 180.0f;

    // compute light pos in cartesian coordinates
    // and scale with maximum intensity
    // phi will effect the amount of we intend to put
    // on a pixel
    float pos[3];
    pos[0] = 255.99f * cos(phi)*cos(theta);
    pos[1] = 255.99f * cos(phi)*sin(theta);
    pos[2] = 255.99f * sin(phi);

    // compute gradient vector
    array gx = convolve(in, vg);
    array gy = convolve(in, hg);

    float pxlz = (6 * 255.0f) / depth;
    array zdepth = constant(pxlz, gx.dims());
    array vdot = gx*pos[0] + gy*pos[1] + pxlz*pos[2];
    array outwd = vdot < 0.0f;
    array norm = vdot / sqrt(gx*gx + gy*gy + zdepth*zdepth);

    array color = outwd * 0.0f + (1 - outwd) * norm;
    return color;
}

int main(int argc, char **argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        array lena = loadImage(ASSETS_DIR "/examples/images/lena.ppm", true);

        array prew_mag, prew_dir;
        array sob_mag, sob_dir;
        array lena1ch = colorSpace(lena, AF_GRAY, AF_RGB);
        prewitt(prew_mag, prew_dir, lena1ch);
        sobelFilter(sob_mag, sob_dir, lena1ch);
        array sprd = spread(lena, 3, 3);
        array hrl = hurl(lena, 10, 1);
        array pckng = pick(lena, 40, 2);
        array difog = dog(lena, 1, 2);
        array bil = bilateral(hrl, 3.0f, 40.0f);
        array mf = medianfilter(hrl, 5, 5);
        array gb = gaussianblur(hrl, 3, 3, 0.8);
        array emb = emboss(lena, 45, 20, 10);

        af::Window wnd("Image Filters Demo");
        std::cout << "Press ESC while the window is in focus to exit" << std::endl;
        while (!wnd.close()) {
            wnd.grid(2, 5);
            wnd(0, 0).image(hrl / 255, "Hurl noise");
            wnd(1, 0).image(gb / 255, "Gaussian blur");
            wnd(0, 1).image(bil / 255, "Bilateral filter on hurl noise");
            wnd(1, 1).image(mf / 255, "Median filter on hurl noise");
            wnd(0, 2).image(prew_mag / 255, "Prewitt edge filter");
            wnd(1, 2).image(sob_mag / 255, "Sobel edge filter");
            wnd(0, 3).image(sprd / 255, "Spread filter");
            wnd(1, 3).image(pckng / 255, "Pick filter");
            wnd(0, 4).image(difog / 255, "Difference of gaussians(3x3 and 5x5)");
            wnd(1, 4).image(emb / 255, "Emboss effect");
            wnd.show();
        }

    }
    catch (af::exception& e) {
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