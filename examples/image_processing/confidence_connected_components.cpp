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

int main(int argc, char* argv[]) {
    try {

        unsigned s[1]       = {132};
        unsigned radius     = 3;
        unsigned multiplier = 3;
        int iter            = 5;

        array A = loadImage(ASSETS_DIR "/examples/images/donut.png", false);

        array seedx(1, 1, s);
        array seedy(1, 1, s);
        array ring =
            confidenceCC(A, seedx, seedy, radius, multiplier, iter, 255);

        seedx = (seedx + 20.0).as(u32);
        seedy = (seedy + 20.0).as(u32);
        array core =
            confidenceCC(A, seedx, seedy, radius, multiplier, iter, 255);

        seedx = (seedx / 10.0).as(u32);
        seedy = (seedy / 10.0).as(u32);
        array background =
            confidenceCC(A, seedx, seedy, radius, multiplier, iter, 255);

        af::Window wnd("Confidence Connected Components demo");
        while(!wnd.close()) {
            wnd.grid(2, 2);
            wnd(0, 0).image(A, "Input");
            wnd(0, 1).image(ring, "Ring Component - Seed(132, 132)");
            wnd(1, 0).image(core, "Center Black Hole - Seed(152, 152)");
            wnd(1, 1).image(background, "Background - Seed(15, 15)");
            wnd.show();
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
