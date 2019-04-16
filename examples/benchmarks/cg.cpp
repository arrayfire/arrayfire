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

using namespace af;

static size_t dimension         = 4 * 1024;
static const int maxIter        = 10;
static const int sparsityFactor = 7;

static array A;
static array spA;  // Sparse A
static array x0;
static array b;

void setupInputs() {
    // Generate a random input: A
    array T = randu(dimension, dimension, f32);
    // Create 0s in input.
    // Anything that is no divisible by sparsityFactor will become 0.
    A = floor(T * 1000);
    A = A * ((A % sparsityFactor) == 0) / 1000;
    // Make it positive definite
    A = transpose(A) + A + A.dims(0) * identity(A.dims(0), A.dims(0), f32);

    // Make A sparse as spA
    spA = sparse(A);

    // Generate x0: Random guess
    x0 = randu(A.dims(0), f32);

    // Generate b
    b = matmul(A, x0);

    std::cout << "Sparsity of A = "
              << 100.f * (float)sparseGetNNZ(spA) / (float)spA.elements() << "%"
              << std::endl;
    std::cout << "Memory Usage of A = " << A.bytes() / (1024.f * 1024.f)
              << " MB" << std::endl;
    std::cout << "Memory Usage of spA = "
              << (sparseGetValues(spA).bytes() + sparseGetRowIdx(spA).bytes() +
                  sparseGetColIdx(spA).bytes()) /
                     (1024.f * 1024.f)
              << " MB" << std::endl;
}

void sparseConjugateGradient(void) {
    array x = constant(0, b.dims(), f32);
    array r = b - matmul(spA, x);
    array p = r;

    for (int i = 0; i < maxIter; ++i) {
        array Ap        = matmul(spA, p);
        array alpha_num = dot(r, r);
        array alpha_den = dot(p, Ap);
        array alpha     = alpha_num / alpha_den;
        r -= tile(alpha, Ap.dims()) * Ap;
        x += tile(alpha, Ap.dims()) * p;
        array beta_num = dot(r, r);
        array beta     = beta_num / alpha_num;
        p              = r + tile(beta, p.dims()) * p;
    }
}

void denseConjugateGradient(void) {
    array x = constant(0, b.dims(), f32);
    array r = b - matmul(A, x);
    array p = r;

    for (int i = 0; i < maxIter; ++i) {
        array Ap        = matmul(A, p);
        array alpha_num = dot(r, r);
        array alpha_den = dot(p, Ap);
        array alpha     = alpha_num / alpha_den;
        r -= tile(alpha, Ap.dims()) * Ap;
        x += tile(alpha, Ap.dims()) * p;
        array beta_num = dot(r, r);
        array beta     = beta_num / alpha_num;
        p              = r + tile(beta, p.dims()) * p;
    }
}

void checkConjugateGradient(const af::array in) {
    array x = constant(0, b.dims(), f32);
    array r = b - matmul(in, x);
    array p = r;

    for (int i = 0; i < maxIter; ++i) {
        array Ap        = matmul(in, p);
        array alpha_num = dot(r, r);
        array alpha_den = dot(p, Ap);
        array alpha     = alpha_num / alpha_den;
        r -= tile(alpha, Ap.dims()) * Ap;
        x += tile(alpha, Ap.dims()) * p;
        array beta_num = dot(r, r);
        array beta     = beta_num / alpha_num;
        p              = r + tile(beta, p.dims()) * p;
    }
    array res = x0 - x;

    std::cout << "Final difference in solutions:\n";
    af_print(dot(res, res));
}

int main(int, char **) {
    af::info();
    setupInputs();

    std::cout << "Verifying Dense Conjugate Gradient:" << std::endl;
    checkConjugateGradient(A);

    std::cout << "Verifying Sparse Conjugate Gradient:" << std::endl;
    checkConjugateGradient(spA);

    af::sync();

    std::cout << "Dense Conjugate Gradient Time: "
              << timeit(denseConjugateGradient) * 1000 << "ms" << std::endl;

    std::cout << "Sparse Conjugate Gradient Time: "
              << timeit(sparseConjugateGradient) * 1000 << "ms" << std::endl;

    return 0;
}
