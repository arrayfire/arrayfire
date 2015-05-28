/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <iostream>
#include <cstdio>

using namespace af;

int main(int argc, char *argv[])
{
    try {
        static const float h_kernel[] = {1, 1, 1, 1, 0, 1, 1, 1, 1};
        static const int reset = 500;
        static const int game_w = 128, game_h = 128;

        af::info();

        std::cout << "This example demonstrates the Conway's Game of Life using ArrayFire" << std::endl
                  << "There are 4 simple rules of Conways's Game of Life" << std::endl
                  << "1. Any live cell with fewer than two live neighbours dies, as if caused by under-population." << std::endl
                  << "2. Any live cell with two or three live neighbours lives on to the next generation." << std::endl
                  << "3. Any live cell with more than three live neighbours dies, as if by overcrowding." << std::endl
                  << "4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction." << std::endl
                  << "Each white block in the visualization represents 1 alive cell, black space represents dead cells" << std::endl
                  ;

        af::Window myWindow(512, 512, "Conway's Game of Life using ArrayFire");

        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array kernel(3, 3, h_kernel, afHost);
        array state;
        state = (af::randu(game_h, game_w, f32) > 0.5).as(f32);

        while(!myWindow.close()) {

            myWindow.image(state);
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

            // Update state
            state = state * C0 + C1;
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

