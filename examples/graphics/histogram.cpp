/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cstdio>
#include <math.h>

using namespace af;

int main(int argc, char *argv[])
{
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(512, 512, "Histogram example using ArrayFire");

        array img = loadimage(ASSETS_DIR"/examples/images/lena.ppm", false);
        array hist_out = histogram(img, 256, 0, 255);

        for (int i = 1; i < 200; i++)
        {
            af::timer delay = timer::start();
            myWindow.hist(hist_out, 0, 255);
            double fps = 15;
            while(timer::stop(delay) < (1 / fps)) { }
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

