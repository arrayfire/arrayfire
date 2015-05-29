/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <arrayfire.h>
#include "../common/progress.h"

using namespace af;

const float h_sx_kernel[] = {  1,  2,  1,
    0,  0,  0,
    -1, -2, -1
};
const float h_sy_kernel[] = { -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};
const float h_lp_kernel[] = { -0.5f, -1.0f, -0.5f,
    -1.0f,  6.0f, -1.0f,
    -0.5f, -1.0f, -0.5f
};

array edges_slice(array x)
{
    array ret;
    static array kernelx = array(dim4(3, 3), h_sx_kernel);
    static array kernely = array(dim4(3, 3), h_sy_kernel);
    ret = convolve(x, kernelx) + convolve(x, kernely);
    return abs(ret);
}

array gauss(array x, float u, float s)
{
    double f = 1 / sqrt(2 * af::Pi * s * s);
    array e = exp(-pow((x - u), 2) / (2 * s * s));
    return f * e;
}

array segment_volume(array A, int k)
{
    array I1 = A(span, span, k);

    float mx = max<float>(I1);
    float mn = min<float>(I1);

    float u0 = 0.9 * mx;
    float s0 = (mx - mn) / 2;

    float u1 = 1.1 * mn;
    float s1 = (mx - mn) / 2;

    array L0  = gauss(I1, u0, s0);
    array L11 = gauss(I1, u1, s1);
    array L10;
    array L12;
    static array kernel = constant(1, 3, 3) / 9;
    static array L11_old;
    static array L12_old;

    if (k == 0) {
        L11 = convolve(L11, kernel);
        L10 = L11;
    } else {
        L10 = L11_old;
        L11 = L12_old;
    }

    if (k < A.dims(2) - 1) {
        L12 = gauss(A(span, span, k + 1), u1, s1);
        L12 = convolve(L12, kernel);
    } else {
        L12 = L11;
    }

    L11_old = L11;
    L12_old = L12;

    array L1 = (L10 + L11 + L12) / 3;
    array S = (L0 > L1);
    return S.as(A.type());
}

void brain_seg(bool console)
{
    af::Window wnd("Brain Segmentation Demo");
    wnd.setColorMap(AF_COLORMAP_HEAT);

    double time_total = 30; // run for N seconds

    array B = loadImage(ASSETS_DIR "/examples/images/brain.png");
    int slices = 256;

    B = moddims(B, dim4(B.dims(0), B.dims(1)/slices, slices));
    af::sync();

    int N = 2 * slices - 1;

    timer t = timer::start();
    int iter = 0;

    /* loop forward and backward for 100 frames
     * exit if the user presses escape or the animation
     * ends
     */
    for (int i = 0; !wnd.close(); i++) {
        iter++;

        int j = i % N;
        int k = std::min(j, N - j);
        array Bi = B(span, span, k);

        /* process */
        array Si = segment_volume(B, k);
        array Ei = edges_slice(Si);
        array Mi = meanShift(Bi, 10, 10, 5);

        /* visualization */
        if (!console) {
            wnd.grid(2, 2);

            wnd(0, 0).image(Bi/255.f, "Input");
            wnd(1, 0).image(Ei, "Edges");
            wnd(0, 1).image(Mi/255.f, "Meanshift");
            wnd(1, 1).image(Si, "Segmented");

            wnd.show();
        } else {
            /* sync the operations so that current
             * iteration comptation finishes
             * */
            af::sync();
        }

        /* we have had ran throuh simlation results
         * exit the rendering loop */
        if (!progress(iter, t, time_total))
            break;
        if (!(i<100*N))
            break;
    }
}

int main(int argc, char* argv[])
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::setDevice(device);
        af::info();

        printf("Brain segmentation example\n");
        brain_seg(console);

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}
