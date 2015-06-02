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

array threshold(const array &in, float thresholdValue)
{
    int channels = in.dims(2);
    array ret_val = in.copy();
    if (channels>1)
        ret_val = colorSpace(in, AF_GRAY, AF_RGB);
    ret_val = (ret_val<thresholdValue)*0.0f + 255.0f*(ret_val>thresholdValue);
    return ret_val;
}

/**
* Note:
* suffix B indicates subset of all graylevels before current gray level
* suffix F indicates subset of all graylevels after current gray level
*/
array otsu(const array& in)
{
    array gray;
    int channels = in.dims(2);
    if (channels>1)
        gray = colorSpace(in, AF_GRAY, AF_RGB);
    else
        gray = in;
    unsigned total = gray.elements();
    array hist = histogram(gray, 256, 0.0f, 255.0f);
    array wts = range(256);

    array wtB = accum(hist);
    array wtF = total - wtB;
    array sumB = accum(wts*hist);
    array meanB = sumB / wtB;
    float lastElemInSumB;
    sumB(seq(255, 255, 1)).host((void*)&lastElemInSumB);
    array meanF = (lastElemInSumB - sumB) / wtF;
    array mDiff = meanB - meanF;

    array interClsVar = wtB * wtF * mDiff * mDiff;

    float max = af::max<float>(interClsVar);
    float threshold2 = where(interClsVar == max).scalar<unsigned>();
    array threshIdx = where(interClsVar >= max);
    float threshold1 = threshIdx.elements()>0 ? threshIdx.scalar<unsigned>() : 0.0f;

    return threshold(gray, (threshold1 + threshold2) / 2.0f);
}

int main(int argc, char **argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        array bimodal = loadImage(ASSETS_DIR "/examples/images/noisy_square.png", false);
        bimodal = resize(0.75f, bimodal);

        array bt = threshold(bimodal, 180.0f);
        array ot = otsu(bimodal);
        array bimodHist = histogram(bimodal, 256, 0, 255);
        array smooth = convolve(bimodal, gaussianKernel(5, 5));
        array smoothHist = histogram(smooth, 256, 0, 255);

        af::Window wnd("Binary Thresholding Algorithms");
        std::cout << "Press ESC while the window is in focus to proceed to exit" << std::endl;
        while (!wnd.close()) {
            wnd.grid(3, 3);
            wnd(0, 0).image(bimodal / 255, "Input Image");
            wnd(1, 0).image(bimodal / 255, "Input Image");
            wnd(2, 0).image(smooth / 255, "Input Smoothed by Gaussian Filter");
            wnd(0, 1).hist(bimodHist, 0, 255, "Input Histogram");
            wnd(1, 1).hist(bimodHist, 0, 255, "Input Histogram");
            wnd(2, 1).hist(smoothHist, 0, 255, "Smoothed Input Histogram");
            wnd(0, 2).image(bt, "Simple Binary threshold");
            wnd(1, 2).image(ot, "Otsu's Threshold");
            wnd(2, 2).image(otsu(smooth), "Otsu's Threshold on Smoothed Image");
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
