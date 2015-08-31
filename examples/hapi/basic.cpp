/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <af/hapi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace af;

std::vector<float> input(100);


// Generate a random number between 0 and 1
// return a uniform number in [0,1].
double unifRand()
{
    return rand() / double(RAND_MAX);
}

void testBackend()
{
    af_info();

    dim_t dims[] = {10, 10, 1, 1};

    af_array A = 0;
    af_array B = 0;

    af_create_array(&A, &(input.front()), 4, dims, af_dtype::f32);
    af_print_array(A);

    af_constant(&B, 0.5, 4, dims, af_dtype::f32);
    af_print_array(B);

    af_release_array(A);
    af_release_array(B);
}

int main(int argc, char *argv[])
{
    std::generate(input.begin(), input.end(), unifRand);

    if (AF_SUCCESS == af_set_backend(AF_BACKEND_CPU))
        testBackend();

    if (AF_SUCCESS == af_set_backend(AF_BACKEND_OPENCL))
        testBackend();

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
    #endif

    return 0;
}
