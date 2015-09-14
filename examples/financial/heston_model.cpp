/**********************************************************************************************
 * Copyright (c) 2015, Michael Nowotny
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used
 * to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***********************************************************************************************/

#include <stdio.h>
#include <iostream>
#include <arrayfire.h>

using namespace std;
using namespace af;

void simulateHestonModel(af::array &xres, af::array &vres,
                         float T, unsigned int N, unsigned int R, float mu, float kappa,
                         float vBar, float sigmaV, float rho, float x0, float v0)
{
    float deltaT = T / (float)(N - 1);

    af::array x[] = {af::constant(x0, R), af::constant(0, R)};
    af::array v[] = {af::constant(v0, R), af::constant(0, R)};

    float sqrtDeltaT = sqrt(deltaT);

    float sqrtOneMinusRhoSquare = sqrt(1 - rho*rho);

    float mArray[] = {rho, sqrtOneMinusRhoSquare};
    af::array m(2, 1, mArray);

    unsigned int tPrevious = 0, tCurrent = 0;
    af::array zeroConstant = constant(0, R);

    for (unsigned int t = 1; t < N; t++) {
        tPrevious = (t+1) % 2;
        tCurrent = t % 2;

        af::array dBt = randn(R, 2) * sqrtDeltaT;
        af::array sqrtVLag = af::sqrt(v[tPrevious]);

        x[tCurrent]= x[tPrevious] + (mu - 0.5 * v[tPrevious]) * deltaT + (sqrtVLag * dBt(span, 0));
        af::array vTmp = v[tPrevious] + kappa * (vBar - v[tPrevious]) * deltaT + sigmaV * (sqrtVLag * matmul(dBt, m));
        v[tCurrent] = max(vTmp, zeroConstant);
    }

    xres = x[tCurrent];
    vres = v[tCurrent];
}

int main()
{
    float T = 1;
    unsigned int nT = 10 * T;
    unsigned int R_first_run = 1000;
    unsigned int R = 20000000;

    float x0 = 0; // initial log stock price
    float v0 = pow(0.087, 2); // initial volatility
    float r = log(1.0319); // risk-free rate
    float rho = -0.82; // instantaneous correlation between Brownian motions
    float sigmaV = 0.14; // variance of volatility
    float kappa = 3.46; // mean reversion speed
    float vBar = 0.008; // mean variance
    float k = log(0.95); // strike price


    // Price European call option
    try {
        af::array x;
        af::array v;

        // first run
        simulateHestonModel(x, v, T, nT, R_first_run, r, kappa, vBar, sigmaV, rho, x0, v0);
        af::sync(); // Ensure the first run is finished

        timer::start();
        simulateHestonModel(x, v, T, nT, R, r, kappa, vBar, sigmaV, rho, x0, v0);
        af::sync();
        cout << "Time in simulation: " << timer::stop() << endl;

        af::array K = exp(constant(k, x.dims()));
        af::array zeroConstant = constant(0, x.dims());
        af::array C_CPU = exp(-r * T) * mean(af::max(af::exp(x) - K, zeroConstant));

        af_print(C_CPU);
        return 0;
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
