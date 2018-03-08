/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/*
   monte-carlo estimation of PI

   algorithm:
   - generate random (x,y) samples uniformly
   - count what percent fell inside (top quarter) of unit circle
*/

#include <benchmark/benchmark.h>
#include <benchmark_macros.h>
#include <arrayfire.h>

#include <cmath>
#include <cstdlib>

const int samples = 20e6;

static void piOnHost(benchmark::State& state)
{
    double pi_estimate = 0.0;

    for (auto _ : state) {
        int count = 0;
        for (int i = 0; i < samples; ++i) {
            float x = float(rand()) / RAND_MAX;
            float y = float(rand()) / RAND_MAX;
            if (sqrt(x*x + y*y) < 1)
                count++;
        }
        pi_estimate = 4.0 * count / samples;
    }

    state.counters["Pi"] = pi_estimate;
}

static void piOnDevice(benchmark::State &state)
{
    double pi_estimate = 0.0;

    af::sync();

    for (auto _ : state) {
        AF_BENCH_TIMER_START();

        af::array x = af::randu(samples, f32);
        af::array y = af::randu(samples, f32);
        pi_estimate = 4.0 * af::sum<float>(af::sqrt(x*x + y*y) < 1) / samples;

        AF_BENCH_TIMER_STOP();
    }

    state.counters["Pi"] = pi_estimate;
}

int main(int argc, char ** argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        benchmark::RegisterBenchmark("Host_PI_Estimate", piOnHost)->
            Unit(benchmark::kMillisecond);
        benchmark::RegisterBenchmark("Device_PI_Estimate", piOnDevice)->
            Unit(benchmark::kMillisecond)->UseManualTime();

        benchmark::Initialize(&argc, argv);
        benchmark::RunSpecifiedBenchmarks();
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
