/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include <arrayfire.h>
#include <af/util.h>
#include <cstdlib>

using namespace af;

int main(int argc, char ** argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        printf("\n=== ArrayFire signed(s32) / unsigned(u32) Integer Example ===\n");

        int h_A[] = {1, 2, 4, -1, 2, 0, 4, 2, 3};
        int h_B[] = {2, 3, -5, 6, 0, 10, -12, 0, 1};
        array A = array(3, 3, h_A);
        array B = array(3, 3, h_B);

        printf("--\nSub-refencing and Sub-assignment\n");
        af_print(A);
        af_print(A.col(0));
        af_print(A.row(0));
        A(0) = 11;
        A(1) = 100;
        af_print(A);
        af_print(B);
        A(1,span) = B(2,span);
        af_print(A);

        printf("--Bit-wise operations\n");
        // Returns an array of type s32
        af_print(A & B);
        af_print(A | B);
        af_print(A ^ B);

        printf("\n--Logical operations\n");
        // Returns an array of type b8
        af_print(A && B);
        af_print(A || B);

        printf("\n--Transpose\n");
        af_print(A);
        af_print(A.T());

        printf("\n--Flip Vertically / Horizontally\n");
        af_print(A);
        af_print(flip(A,0));
        af_print(flip(A,1));

        printf("\n--Sum along columns\n");
        af_print(A);
        af_print(sum(A));

        printf("\n--Product along columns\n");
        af_print(A);
        af_print(product(A));

        printf("\n--Minimum along columns\n");
        af_print(A);
        af_print(min(A));

        printf("\n--Maximum along columns\n");
        af_print(A);
        af_print(max(A));

        printf("\n--Minimum along columns with index\n");
        af_print(A);

        array out, idx;
        min(out, idx, A);
        af_print(out);
        af_print(idx);

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
