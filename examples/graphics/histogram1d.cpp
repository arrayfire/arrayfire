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
        af::initGraphics(0);

        float input[] = {1, 2, 1, 1, 3, 6, 7, 8, 3};
        int nbins = 10;
        size_t nElems = sizeof(input)/sizeof(float);
        array hist_in(nElems, input);
        array hist_out = histogram(hist_in, nbins, 0, 9);

        af_print(hist_out);
//        array Y(size,1,y,af::afHost);

       for (int i = 1; i < 200; i++)
       {
           af::timer delay = timer::start();
           histogram1d(hist_in, nbins, 0, 9);
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

