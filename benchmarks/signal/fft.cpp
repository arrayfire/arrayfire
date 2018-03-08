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

static void customArgs(benchmark::internal::Benchmark* b)
{
    for (int i = 7; i<=12; ++i)
        b->Args({i, 1<<i});
}

static void fft2(benchmark::State& state)
{
    double timeInSecs = 0.0;

    const int M = state.range(0);
    const int N = state.range(1);

    const af::array A = af::randu(N, N);

    af::sync();

    for (auto _ : state) {
        AF_BENCH_TIMER_START();

        af::array B = af::fft2(A);

        AF_BENCH_TIMER_STOP();

        //Following should come after AF_BENCH_TIMER_STOP
        //As it uses variable(elapsed_seconds) declared by it
        timeInSecs = elapsed_seconds.count();
    }

    state.counters["Gflops"] = 10.0 * N * N * M / (timeInSecs * 1e9);
}

BENCHMARK(fft2)->Apply(customArgs)->
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
