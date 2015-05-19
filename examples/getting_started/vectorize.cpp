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

using namespace af;

array A, B;

static array dist_naive(array a, array b)
{
    array dist_mat = constant(0, a.dims(1), (int)b.dims(1));

    // Iterate through columns a
    for (int ii = 0; ii < (int)a.dims(1); ii++) {

        // Iterate through columns of b
        for (int jj = 0; jj < (int)b.dims(1); jj++) {

            // Get the sum of absolute differences
            for (int kk = 0; kk < (int)a.dims(0); kk++) {
                dist_mat(ii, jj) += abs(a(kk, ii) - b(kk, jj));
            }
        }
    }

    return dist_mat;
}

static array dist_vec(array a, array b)
{
    array dist_mat = constant(0, (int)a.dims(1), (int)b.dims(1));

    // Iterate through columns a
    for (int ii = 0; ii < (int)a.dims(1); ii++) {
        array avec = a(span, ii);

        // Iterate through columns of b
        for (int jj = 0; jj < (int)b.dims(1); jj++) {
            array bvec = b(span, jj);

            // get SAD using sum on the vector
            dist_mat(ii, jj) = sum(abs(avec - bvec));
        }
    }

    return dist_mat;
}

static array dist_gfor1(array a, array b)
{
    array dist_mat = constant(0, (int)a.dims(1), (int)b.dims(1));

    // GFOR along columns of a
    gfor (seq ii, (int)a.dims(1)) {
        array avec = a(span, ii);

        // Itere through columns of b
        for (int jj = 0; jj < (int)b.dims(1); jj++) {
            array bvec = b(span, jj);

            // get SAD using sum on the vector
            dist_mat(ii, jj) = sum(abs(avec - bvec));
        }
    }

    return dist_mat;
}

static array dist_gfor2(array a, array b)
{
    array dist_mat = constant(0, (int)a.dims(1), (int)b.dims(1));

    // GFOR along columns of b
    gfor (seq jj, (int)b.dims(1)) {
        array bvec = b(span, jj);

        // Iterate through columns of A
        for (int ii = 0; ii < (int)a.dims(1); ii++) {
            array avec = a(span, ii);

            // get SAD using sum on the vector
            dist_mat(ii, jj) = sum(abs(avec - bvec));
        }
    }

    return dist_mat;
}

static array dist_tile1(array a, array b)
{
    // int feat_len = (int)a.dims(0); // Same as (int)b.dims(0);
    int alen = (int)a.dims(1);
    int blen = (int)b.dims(1);

    array dist_mat = constant(0, alen, blen);

    // Iterate through columns of b
    for (int jj = 0; jj < blen; jj++) {

        // Get the column vector of b
        // shape of bvec is (feat_len, 1)
        array bvec = b(span, jj);

        // Tile avec to be same size as a
        // shape of bvec_tiled is (feat_len, alen)
        array bvec_tiled = tile(bvec, 1, alen);

        // Get the sum of absolute differences
        array sad = sum(abs(bvec_tiled - a));

        // sad is row vector, dist_mat needs column vector
        // transpose sad and fill in dist_mat
        dist_mat(span, jj) = sad.T();
    }

    return dist_mat;
}

static array dist_tile2(array a, array b)
{
    int feat_len = (int)a.dims(0);
    int alen = (int)a.dims(1);
    int blen = (int)b.dims(1);

    // Shape of a is (feat_len, alen, 1)
    array a_mod = a;
    // Reshape b from (feat_len, blen) to (feat_len, 1, blen)
    array b_mod = moddims(b, feat_len, 1, blen);

    // Tile both matrices to be (feat_len, alen, blen)
    array a_tiled = tile(a_mod, 1, 1, blen);
    array b_tiled = tile(b_mod, 1, alen, 1);

    // Do The sum operation along first dimension
    // Output is of shape (1, alen, blen)
    array dist_mod = sum(abs(a_tiled - b_tiled));

    // Reshape dist_mat from (1, alen, blen) to (alen, blen)
    array dist_mat = moddims(dist_mod, alen, blen);
    return dist_mat;
}

static void bench_naive()
{
    dist_naive(A, B);
}

static void bench_vec()
{
    dist_vec(A, B);
}

static void bench_gfor1()
{
    dist_gfor1(A, B);
}

static void bench_gfor2()
{
    dist_gfor2(A, B);
}

static void bench_tile1()
{
    dist_tile1(A, B);
}

static void bench_tile2()
{
    dist_tile2(A, B);
}

int main(int argc, char **argv)
{
    try {

        af::info();

        // Do not increase the sizes
        // dist_naive and dist_vec get too slow at large sizes
        A = randu(3, 200);
        B = randu(3, 300);

        array d1 = dist_naive(A, B);
        array d2 = dist_vec  (A, B);
        array d3 = dist_gfor1(A, B);
        array d4 = dist_gfor2(A, B);
        array d5 = dist_tile1(A, B);
        array d6 = dist_tile2(A, B);

        printf("Max. Error for dist_vec  : %f\n", max<float>(abs(d1 - d2)));
        printf("Max. Error for dist_gfor1: %f\n", max<float>(abs(d1 - d3)));
        printf("Max. Error for dist_gfor2: %f\n", max<float>(abs(d1 - d4)));
        printf("Max. Error for dist_tile1: %f\n", max<float>(abs(d1 - d5)));
        printf("Max. Error for dist_tile2: %f\n", max<float>(abs(d1 - d6)));

        printf("\n");

        printf("Time for dist_naive: %2.2fms\n", 1000 * timeit(bench_naive));
        printf("Time for dist_vec  : %2.2fms\n", 1000 * timeit(bench_vec  ));
        printf("Time for dist_gfor1: %2.2fms\n", 1000 * timeit(bench_gfor1));
        printf("Time for dist_gfor2: %2.2fms\n", 1000 * timeit(bench_gfor2));
        printf("Time for dist_tile1: %2.2fms\n", 1000 * timeit(bench_tile1));
        printf("Time for dist_tile2: %2.2fms\n", 1000 * timeit(bench_tile2));

    } catch(af::exception ex) {
        fprintf(stderr, "%s\n", ex.what());
        throw;
    }

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
    #endif
}
