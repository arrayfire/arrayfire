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

typedef enum {
    MEAN = 0,
    MEDIAN,
    MINMAX_AVG
} LocalThresholdType;

array threshold(const array &in, float thresholdValue)
{
    int channels = in.dims(2);
    array ret_val = in.copy();
    if (channels>1)
        ret_val = colorSpace(in, AF_GRAY, AF_RGB);
    ret_val = (ret_val<thresholdValue)*0.0f + 255.0f*(ret_val>thresholdValue);
    return ret_val;
}

array adaptiveThreshold(const array &in, LocalThresholdType kind, int window_size, int constnt)
{
    int wr = window_size;
    array ret_val = colorSpace(in, AF_GRAY, AF_RGB);
    if (kind == MEAN) {
        array wind = constant(1, wr, wr) / (wr*wr);
        array mean = convolve(ret_val, wind);
        array diff = mean - ret_val;
        ret_val = (diff<constnt)*0.f + 255.f*(diff>constnt);
    }
    else if (kind == MEDIAN) {
        array medf = medfilt(ret_val, wr, wr);
        array diff = medf - ret_val;
        ret_val = (diff<constnt)*0.f + 255.f*(diff>constnt);
    }
    else if (kind == MINMAX_AVG) {
        array minf = minfilt(ret_val, wr, wr);
        array maxf = maxfilt(ret_val, wr, wr);
        array mean = (minf + maxf) / 2.0f;
        array diff = mean - ret_val;
        ret_val = (diff<constnt)*0.f + 255.f*(diff>constnt);
    }
    ret_val = 255.f - ret_val;
    return ret_val;
}

array iterativeThreshold(const array &in)
{
    array ret_val = colorSpace(in, AF_GRAY, AF_RGB);
    float T = mean<float>(ret_val);
    bool isContinue = true;
    while (isContinue) {
        array region1 = (ret_val > T)*ret_val;
        array region2 = (ret_val <= T)*ret_val;
        float r1_avg = mean<float>(region1);
        float r2_avg = mean<float>(region2);
        float tempT = (r1_avg + r2_avg) / 2.0f;
        if (abs(tempT - T)<0.01f) {
            break;
        }
        T = tempT;
    }
    return threshold(ret_val, T);
}

int main(int argc, char **argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        array sudoku = loadImage(ASSETS_DIR "/examples/images/sudoku.jpg", true);

        array mnt = adaptiveThreshold(sudoku, MEAN, 37, 10);
        array mdt = adaptiveThreshold(sudoku, MEDIAN, 7, 4);
        array mmt = adaptiveThreshold(sudoku, MINMAX_AVG, 11, 4);
        array itt = 255.0f - iterativeThreshold(sudoku);

        af::Window wnd("Adaptive Thresholding Algorithms");
        std::cout << "Press ESC while the window is in focus to exit" << std::endl;
        while (!wnd.close()) {
            wnd.grid(2, 3);
            wnd(0, 0).image(sudoku / 255, "Input");
            wnd(1, 0).image(mnt, "Adap. Threshold(Mean)");
            wnd(0, 1).image(mdt, "Adap. Threshold(Median)");
            wnd(1, 1).image(mmt, "Adap. Threshold(Avg. Min,Max)");
            wnd(0, 2).image(itt, "Iterative Threshold");
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