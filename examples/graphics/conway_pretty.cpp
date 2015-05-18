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

using namespace af;

int main(int argc, char *argv[])
{
    try {
        static const float h_kernel[] = {1, 1, 1, 1, 0, 1, 1, 1, 1};
        static const int reset = 500;
        static const int game_w = 100, game_h = 100;

        af::info();
        af::Window myWindow(512, 512, "Conway's Game Of Life");
        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array kernel(3, 3, h_kernel, af::afHost);
        array state;
        state = (af::randu(game_h, game_w, f32) > 0.4).as(f32);

        array display = tile(state, 1, 1, 3, 1);

        while(frame_count <= 1500) {
            af::timer delay = timer::start();

            myWindow.image(display);
            frame_count++;

            // Generate a random starting state
            if(frame_count % reset == 0)
                state = (af::randu(game_h, game_w, f32) > 0.5).as(f32);

            // Convolve gets neighbors
            af::array nHood = convolve(state, kernel);

            // Generate conditions for life
            // state == 1 && nHood < 2 ->> state = 0
            // state == 1 && nHood > 3 ->> state = 0
            // else if state == 1 ->> state = 1
            // state == 0 && nHood == 3 ->> state = 1
            af::array C0 = (nHood == 2);
            af::array C1 = (nHood == 3);

            array a0 = (state == 1) && (nHood < 2); // Die of under population
            array a1 = (state != 0) && (C0 || C1);         // Continue to live
            array a2 = (state == 0) && C1;          // Reproduction
            array a3 = (state == 1) && (nHood > 3); // Over-population

            display = join(2, join(2, a0 + a1, a1 + a2), a3).as(f32);

            // Update state
            state = state * C0 + C1;

            double fps = 5;
            while(timer::stop(delay) < (1 / fps)) { }
        }
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

