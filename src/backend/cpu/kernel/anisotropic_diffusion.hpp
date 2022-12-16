/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <math.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

using std::exp;
using std::pow;
using std::sqrt;

namespace arrayfire {
namespace cpu {
namespace kernel {

int index(int x, int y, int stride1) { return y * stride1 + x; }

float quad(float value) { return 1.0f / (1.0f + value); }

float computeGradientBasedUpdate(const float mct, const float NW, const float N,
                                 const float NE, const float W, const float C,
                                 const float E, const float SW, const float S,
                                 const float SE,
                                 const af_flux_function fftype) {
    float delta = 0.f;

    float dx, dy, df, db, cx, cxd;

    // centralized derivatives
    dx = (E - W) * 0.5f;
    dy = (S - N) * 0.5f;

    // half-d's and conductance along first dimension
    df = E - C;
    db = C - W;

    float gmsqf = (df * df + 0.25f * pow(dy + 0.5f * (SE - NE), 2.f)) * mct;
    float gmsqb = (db * db + 0.25f * pow(dy + 0.5f * (SW - NW), 2.f)) * mct;
    if (fftype == AF_FLUX_EXPONENTIAL) {
        cx  = exp(gmsqf);
        cxd = exp(gmsqb);
    } else {
        cx  = quad(gmsqf);
        cxd = quad(gmsqb);
    }
    delta = (cx * df - cxd * db);

    // half-d's and conductance along second dimension
    df = S - C;
    db = C - N;

    gmsqf = (df * df + 0.25f * pow(dx + 0.5f * (SE - SW), 2.f)) * mct;
    gmsqb = (db * db + 0.25f * pow(dx + 0.5f * (NE - NW), 2.f)) * mct;
    if (fftype == AF_FLUX_EXPONENTIAL) {
        cx  = exp(gmsqf);
        cxd = exp(gmsqb);
    } else {
        cx  = quad(gmsqf);
        cxd = quad(gmsqb);
    }
    delta += (cx * df - cxd * db);

    return delta;
}

float computeCurvatureBasedUpdate(const float mct, const float NW,
                                  const float N, const float NE, const float W,
                                  const float C, const float E, const float SW,
                                  const float S, const float SE) {
    float delta     = 0.f;
    float prop_grad = 0.f;

    float df0, db0;
    float dx, dy, df, db, cx, cxd, gmf, gmb, gmsqf, gmsqb;

    // centralized derivatives
    dx = (E - W) * 0.5f;
    dy = (S - N) * 0.5f;

    // half-d's and conductance along first dimension
    df  = E - C;
    db  = C - W;
    df0 = df;
    db0 = db;

    gmsqf = (df * df + 0.25f * pow(dy + 0.5f * (SE - NE), 2.f));
    gmsqb = (db * db + 0.25f * pow(dy + 0.5f * (SW - NW), 2.f));

    gmf = sqrt(1.0e-10f + gmsqf);
    gmb = sqrt(1.0e-10f + gmsqb);

    cx  = exp(gmsqf * mct);
    cxd = exp(gmsqb * mct);

    delta = ((df / gmf) * cx - (db / gmb) * cxd);

    // half-d's and conductance along second dimension
    df = S - C;
    db = C - N;

    gmsqf = (df * df + 0.25f * pow(dx + 0.5f * (SE - SW), 2.f));
    gmsqb = (db * db + 0.25f * pow(dx + 0.5f * (NE - NW), 2.f));
    gmf   = sqrt(1.0e-10f + gmsqf);
    gmb   = sqrt(1.0e-10f + gmsqb);

    cx  = exp(gmsqf * mct);
    cxd = exp(gmsqb * mct);

    delta += ((df / gmf) * cx - (db / gmb) * cxd);

    if (delta > 0.f) {
        prop_grad +=
            (pow(fminf(db0, 0.0f), 2.0f) + pow(fmaxf(df0, 0.0f), 2.0f));
        prop_grad += (pow(fminf(db, 0.0f), 2.0f) + pow(fmaxf(df, 0.0f), 2.0f));
    } else {
        prop_grad +=
            (pow(fmaxf(db0, 0.0f), 2.0f) + pow(fminf(df0, 0.0f), 2.0f));
        prop_grad += (pow(fmaxf(db, 0.0f), 2.0f) + pow(fminf(df, 0.0f), 2.0f));
    }

    return sqrt(prop_grad) * delta;
}

template<typename T, bool isMCDE>
void anisotropicDiffusion(Param<T> inout, const float dt, const float mct,
                          const af_flux_function fftype) {
    const auto dims     = inout.dims();
    const auto strides  = inout.strides();
    const auto d1stride = strides[1];
    const int d0        = dims[0] - 1;
    const int d1        = dims[1] - 1;
    const int d2        = dims[2];
    const int d3        = dims[3];

    for (int b3 = 0; b3 < d3; ++b3) {
        for (int b2 = 0; b2 < d2; ++b2) {
            T* img = inout.get() + b2 * strides[2] + b3 * strides[3];
            for (int j = 1; j < d1; ++j) {
                for (int i = 1; i < d0; ++i) {
                    float C     = 0.f;
                    float delta = 0.f;

                    const int ip1 = i + 1;
                    const int im1 = i - 1;
                    const int jp1 = j + 1;
                    const int jm1 = j - 1;

                    if (isMCDE) {
                        delta = computeCurvatureBasedUpdate(
                            mct, img[index(im1, jm1, d1stride)],
                            img[index(i, jm1, d1stride)],
                            img[index(ip1, jm1, d1stride)],
                            img[index(im1, j, d1stride)],
                            C = img[index(i, j, d1stride)],
                            img[index(ip1, j, d1stride)],
                            img[index(im1, jp1, d1stride)],
                            img[index(i, jp1, d1stride)],
                            img[index(ip1, jp1, d1stride)]);

                    } else {
                        delta = computeGradientBasedUpdate(
                            mct, img[index(im1, jm1, d1stride)],
                            img[index(i, jm1, d1stride)],
                            img[index(ip1, jm1, d1stride)],
                            img[index(im1, j, d1stride)],
                            C = img[index(i, j, d1stride)],
                            img[index(ip1, j, d1stride)],
                            img[index(im1, jp1, d1stride)],
                            img[index(i, jp1, d1stride)],
                            img[index(ip1, jp1, d1stride)], fftype);
                    }

                    img[i + j * d1stride] = (T)(C + delta * dt);
                }
            }
        }
    }
}
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
