/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <algorithm>
#include <cstdio>
#include <vector>

using namespace af;

std::vector<float> input(100);

// Generate a random number between 0 and 1
// return a uniform number in [0,1].
double unifRand() { return rand() / double(RAND_MAX); }

void testBackend() {
    af::info();

    af::dim4 dims(10, 10, 1, 1);

    af::array A(dims, &input.front());
    af_print(A);

    af::array B = af::constant(0.5, dims, f32);
    af_print(B);
}

int main(int, char**) {
    std::generate(input.begin(), input.end(), unifRand);

    try {
        printf("Trying CPU Backend\n");
        af::setBackend(AF_BACKEND_CPU);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying CPU backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("Trying CUDA Backend\n");
        af::setBackend(AF_BACKEND_CUDA);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying CUDA backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("Trying OpenCL Backend\n");
        af::setBackend(AF_BACKEND_OPENCL);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying OpenCL backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    return 0;
}
