/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace af;

array normalize01(const array& in) {
    float min = af::min<float>(in);
    float max = af::max<float>(in);
    return (in - min) / (max - min);
}

void markCrossHair(array& in, const unsigned x, const unsigned y,
                   const float val) {
    const int draw_len = 5;
    for (int i = -1; i < 2; i++) {
        in(x + i, seq(y - draw_len, y + draw_len), 0) = val;
        in(x + i, seq(y - draw_len, y + draw_len), 1) = 0.f;
        in(x + i, seq(y - draw_len, y + draw_len), 2) = 0.f;

        in(seq(x - draw_len, x + draw_len), y + i, 0) = val;
        in(seq(x - draw_len, x + draw_len), y + i, 1) = 0.f;
        in(seq(x - draw_len, x + draw_len), y + i, 2) = 0.f;
    }
}

int main(int argc, char* argv[]) {
    try {
        unsigned radius     = 3;
        unsigned multiplier = 2;
        int iter            = 3;

        array input =
            loadImage(ASSETS_DIR "/examples/images/depression.jpg", false);
        array normIn = normalize01(input);

        unsigned seedx = 162;
        unsigned seedy = 126;
        array blob = confidenceCC(input, 1, &seedx, &seedy, radius, multiplier,
                                  iter, 255);

        array colorIn  = colorSpace(normIn, AF_RGB, AF_GRAY);
        array colorOut = colorSpace(blob, AF_RGB, AF_GRAY);

        markCrossHair(colorIn, seedx, seedy, 1);
        markCrossHair(colorOut, seedx, seedy, 255);

        af::Window wnd("Confidence Connected Components Demo");
        while (!wnd.close()) {
            wnd.grid(1, 2);
            wnd(0, 0).image(colorIn, "Input Brain Scan");
            wnd(0, 1).image(colorOut, "Region connected to Seed(162, 126)");
            wnd.show();
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
