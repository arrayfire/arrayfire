/*******************************************************
* Copyright (c) 2014, ArrayFire
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

/**
* contrast value should be in the rnage [-1,1]
* */
array changeContrast(const array &in, const float contrast)
{
    float scale = tan((contrast + 1)*Pi / 4);
    return (((in / 255.0f - 0.5f) * scale + 0.5f) * 255.0f);
}

/**
* brightness value should be in the rnage [0,1]
* */
array changeBrightness(const array &in, const float brightness, const float channelMax = 255.0f)
{
    float factor = brightness*channelMax;
    return (in + factor);
}

array clamp(const array &in, float min = 0.0f, float max = 255.0f)
{
    return ((in<min)*0.0f + (in>max)*255.0f + (in >= min && in <= max)*in);
}

/**
* radius effects the level of details that will effected during sharpening process
* amount value should be in the range [0,1] or [1,]
* note: value of 1.0 for amount results unsharp masking
*       values > 1.0 results in highboost filter effect
* */
array usm(const array &in, float radius, float amount)
{
    int gKernelLen = 2 * radius + 1;
    array blurKernel = gaussianKernel(gKernelLen, gKernelLen);
    array blur = convolve(in, blurKernel);
    return (in + amount*(in - blur));
}

/**
* x,y - starting position of zoom
* width, height - dimensions of the rectangle to where we have to zoom in
* */
array digZoom(const array &in, int x, int y, int width, int height)
{
    array cropped = in(seq(x, width - 1), seq(y, height - 1), span);
    return resize(cropped, (unsigned)in.dims(0), (unsigned)in.dims(1));
}

/**
* a - foregound image
* b - background image
* mask - mask map
* */
array alphaBlend(const array &a, const array &b, const array &mask)
{
    array tiledMask;
    if (mask.dims(2) != a.dims(2))
        tiledMask = tile(mask, 1, 1, a.dims(2));
    return a*tiledMask + (1.0f - tiledMask)*b;
}

void normalizeImage(array &in)
{
    float min = af::min<float>(in);
    float max = af::max<float>(in);
    in = 255.0f*((in - min) / (max - min));
}

/**
* dimensions of the mask control the thickness of the boundary that
* will be extracted by the following function
*/
array boundary(const array &in, const array &mask)
{
    array ret_val = in - erode(in, mask);
    normalizeImage(ret_val);
    return ret_val;
}

int main(int argc, char **argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        array man = loadImage(ASSETS_DIR "/examples/images/man.jpg", true);
        array fight = loadImage(ASSETS_DIR "/examples/images/fight.jpg", true);
        array nature = resize(loadImage(ASSETS_DIR "/examples/images/nature.jpg", true), fight.dims(0), fight.dims(1));

        array intensity = colorSpace(fight, AF_GRAY, AF_RGB);
        array mask = clamp(intensity, 10.0f, 255.0f)>0.0f;
        array blend = alphaBlend(fight, nature, mask);
        array highcon = changeContrast(man, 0.3);
        array highbright = changeBrightness(man, 0.2);
        array translated = translate(man, 100, 100, 200, 126);
        array sharp = usm(man, 3, 1.2);
        array zoom = digZoom(man, 28, 10, 192, 192);
        array morph_mask = constant(1, 3, 3);
        array bdry = boundary(man, morph_mask);

        af::Window wnd("Image Editing Operations");
        std::cout << "Press ESC while the window is in focus to exit" << std::endl;
        while (!wnd.close()) {
            wnd.grid(2, 5);
            wnd(0, 0).image(man / 255, "Input");
            wnd(1, 0).image(highcon / 255, "High Contrast");
            wnd(0, 1).image(highbright / 255, "High Brightness");
            wnd(1, 1).image(translated / 255, "Translation");
            wnd(0, 2).image(sharp / 255, "Unsharp Masking");
            wnd(1, 2).image(zoom / 255, "Digital Zoom");
            wnd(0, 3).image(nature / 255, "Background for blend");
            wnd(1, 3).image(fight / 255, "Foreground for blend");
            wnd(0, 4).image(blend / 255, "Alpha blend");
            wnd(1, 4).image(bdry / 255, "Boundary extraction");
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
