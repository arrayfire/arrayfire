/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <arrayfire.h>
#include <af/util.h>

using namespace af;
template<class ty> dtype get_dtype();

template<> dtype get_dtype<float>() { return f32; }
template<> dtype get_dtype<double>() { return f64; }

template<class ty, bool use_barrier>
static ty monte_carlo_barrier(int N, ty K, ty t, ty vol, ty r, ty strike, int steps, ty B)
{
    dtype pres = get_dtype<ty>();
    array payoff = constant(0, N, 1, pres);

    ty dt = t / (ty)(steps - 1);
    array s = constant(strike, N, 1, pres);

    array randmat = randn(N, steps - 1, pres);
    randmat = exp((r - (vol * vol * 0.5)) * dt + vol * sqrt(dt) * randmat);

    array S = product(join(1, s, randmat), 1);

    if (use_barrier) {
        S = S * allTrue(S < B, 1);
    }

    payoff = max(0.0, S - K);
    ty P = mean<ty>(payoff) * exp(-r * t);
    return P;
}

template<class ty, bool use_barrier>
double monte_carlo_bench(int N)
{
    int steps = 180;
    ty stock_price = 100.0;
    ty maturity = 0.5;
    ty volatility = .30;
    ty rate = .01;
    ty strike = 100;
    ty barrier = 115.0;

    timer::start();
    for (int i = 0; i < 10; i++) {
        monte_carlo_barrier<ty, use_barrier>(N, stock_price, maturity, volatility,
                                             rate, strike, steps, barrier);
    }
    return timer::stop() / 10;
}

int main()
{
    try {

        // Warm up and caching
        monte_carlo_bench<float, false>(1000);
        monte_carlo_bench<float, true>(1000);

        for (int n = 10000; n <= 100000; n += 10000) {
            printf("Time for %7d paths - "
                   "vanilla method: %4.3f ms,  "
                   "barrier method: %4.3f ms\n", n,
                   1000 * monte_carlo_bench<float, false>(n),
                   1000 * monte_carlo_bench<float, true>(n));
        }
    } catch (af::exception &ae) {
        std::cerr << ae.what() << std::endl;
    }

    return 0;
}
