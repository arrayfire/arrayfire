/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <benchmark/benchmark.h>
#include <benchmark_macros.h>
#include <arrayfire.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace af;

static size_t dimension = 4 * 1024;
static const int sparsityFactor = 7;
static const int maxIter = 10;

void setupInputs(array& A, array& b, array& spA, array& x0)
{
    // Generate a random input: A
    array T = randu(dimension, dimension, f32);
    // Create 0s in input.
    // Anything that is no divisible by sparsityFactor will become 0.
    A = floor(T * 1000);
    A = A * ((A % sparsityFactor) == 0) / 1000;
    // Make it positive definite
    A = transpose(A) + A + A.dims(0)*identity(A.dims(0), A.dims(0), f32);

    // Make A sparse as spA
    spA = sparse(A);

    // Generate x0: Random guess
    x0 = randu(A.dims(0), f32);

    //Generate b
    b = matmul(A, x0);
}

static void DenseConjugateGradient(benchmark::State& state)
{
    array A, b;
    array spA, x0; // Not used in dense conjugate gradient

    setupInputs(A, b, spA, x0);
    af::sync();

    for (auto _ : state) {
        AF_BENCH_TIMER_START();

        array x = constant(0, b.dims(), f32);
        array r = b - matmul(A, x);
        array p = r;

        for (int i = 0; i < maxIter; ++i) {
            array Ap = matmul(A, p);
            array alpha_num = dot(r, r);
            array alpha_den = dot(p, Ap);
            array alpha = alpha_num/alpha_den;
            r -= tile(alpha, Ap.dims())*Ap;
            x += tile(alpha, Ap.dims())*p;
            array beta_num = dot(r, r);
            array beta = beta_num/alpha_num;
            p = r + tile(beta, p.dims()) * p;
        }

        AF_BENCH_TIMER_STOP();
    }
}

static void SparseConjugateGradient(benchmark::State& state)
{
    array A, b;
    array spA, x0;

    setupInputs(A, b, spA, x0);
    af::sync();

    for (auto _ : state) {
        AF_BENCH_TIMER_START();

        array x = constant(0, b.dims(), f32);
        array r = b - matmul(spA, x);
        array p = r;

        for (int i = 0; i < maxIter; ++i) {
            array Ap = matmul(spA, p);
            array alpha_num = dot(r, r);
            array alpha_den = dot(p, Ap);
            array alpha = alpha_num/alpha_den;
            r -= tile(alpha, Ap.dims())*Ap;
            x += tile(alpha, Ap.dims())*p;
            array beta_num = dot(r, r);
            array beta = beta_num/alpha_num;
            p = r + tile(beta, p.dims()) * p;
        }

        AF_BENCH_TIMER_STOP();
    }
}

BENCHMARK(DenseConjugateGradient)->Unit(benchmark::kMillisecond)->
    UseManualTime();

BENCHMARK(SparseConjugateGradient)->Unit(benchmark::kMillisecond)->
    UseManualTime();

int main(int argc, char ** argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        benchmark::Initialize(&argc, argv);
        benchmark::RunSpecifiedBenchmarks();
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
