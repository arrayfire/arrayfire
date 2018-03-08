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

static void matmul(benchmark::State& state)
{
    double timeInSecs = 0.0;

    const int n = state.range(0);

    const af::array A = af::constant(1, n, n);

    af::sync();

    for (auto _ : state) {
        AF_BENCH_TIMER_START();

        af::array B = af::matmul(A, A);

        AF_BENCH_TIMER_STOP();

        //Following should come after AF_BENCH_TIMER_STOP
        //As it uses variable(elapsed_seconds) declared by it
        timeInSecs = elapsed_seconds.count();
    }

    state.counters["Gflops"] = 2.0 * powf(n,3) / (timeInSecs * 1e9);
}

BENCHMARK(matmul)->DenseRange(128, 2048, 128)->
    Unit(benchmark::kMicrosecond)->UseManualTime();

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
