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
    array payoff = constant(0, 1, N, pres);

    ty dt = t / (ty)(steps - 1);
    array s = strike * constant(1, 1, N, pres);

    array randmat = randn(steps - 1, N, pres);
    randmat = exp((r - (vol * vol * 0.5)) * dt + vol * sqrt(dt) * randmat);

    // FIXME: Change when "mul" is implemented
#if 1
    array S = exp(sum(log(join(0, s, randmat))));
#else
    array S = mul(join(0, s, randmat));
#endif

    if (use_barrier) {
        S = S * alltrue(S < B);
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

        for (int n = 25000; n <= 250000; n += 25000) {
            printf("Time for %7d paths - "
                   "vanilla method: %4.3f ms,  "
                   "barrier method: %4.3f ms\n", n,
                   1000 * monte_carlo_bench<float, false>(n),
                   1000 * monte_carlo_bench<float, true>(n));
        }
    } catch (af::exception &ae) {
        std::cout << ae.what() << std::endl;
    }

    return 0;
}
