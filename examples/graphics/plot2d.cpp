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

        fg_window_handle window;
        fg_plot_handle plot;
        fg_create_window(&window, 640, 480, "Plot", FG_RED, GL_FLOAT);
        fg_plot_init(&plot, window, 640, 480);

        for (int i = 1; i < 200; i++)
        {
            af::timer delay = timer::start();
            array X = range(i)/i;
            array Y = sin(X*af::Pi).as(f32);
            drawPlot(X, Y, plot);
            double fps = 5;
            while(timer::stop(delay) < (1 / fps)) { }
        }

        fg_destroy_plot(plot);
        fg_destroy_window(window);

    } catch (af::exception& e) {
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

