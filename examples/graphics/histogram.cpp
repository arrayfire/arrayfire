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
        af::Window imgWnd("Input Image");

        array img = loadImage(ASSETS_DIR"/examples/images/arrow.jpg", false);
        array hist_out = histogram(img, 256, 0, 255);

        while (!myWindow.close() && !imgWnd.close()) {
            myWindow.hist(hist_out, 0, 255);
            imgWnd.image(img.as(u8));
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

