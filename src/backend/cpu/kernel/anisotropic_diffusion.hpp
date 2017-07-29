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

#include <cassert>
#include <cmath>
#include <algorithm>

using std::exp;
using std::pow;
using std::sqrt;

namespace cpu
{
namespace kernel
{

int index(int x, int y, int stride1)
{
    return x+ y*stride1;
}

float quad(float value)
{
  return 1.0f/(1.0f+value);
}

float computeGradientBasedUpdate(const float mct,
                                 const float NW, const float N, const float NE,
                                 const float  W, const float C, const float  E,
                                 const float SW, const float S, const float SE, const af_flux_function fftype)
{
    float delta = 0.f;

    float dx, dy, df, db, cx, cxd;

    // centralized derivatives
    dx = (E-W)*0.5f;
    dy = (S-N)*0.5f;

    // half-d's and conductance along first dimension
    df  = E - C;
    db  = C - W;

    if (fftype==AF_FLUX_EXPONENTIAL) {
        cx  = exp( (df*df + 0.25f*pow(dy+0.5f*(SE - NE), 2.f)) * mct );
        cxd = exp( (db*db + 0.25f*pow(dy+0.5f*(SW - NW), 2.f)) * mct );
    } else {
        cx  = quad( (df*df + 0.25f*pow(dy+0.5f*(SE - NE), 2.f)) * mct );
        cxd = quad( (db*db + 0.25f*pow(dy+0.5f*(SW - NW), 2.f)) * mct );
    }
    delta += (cx*df - cxd*db);

    // half-d's and conductance along second dimension
    df  = S - C;
    db  = C - N;

    if (fftype==AF_FLUX_EXPONENTIAL) {
        cx  = exp( (df*df + 0.25f*pow(dx+0.5f*(SE - SW), 2.f)) * mct );
        cxd = exp( (db*db + 0.25f*pow(dx+0.5f*(NE - NW), 2.f)) * mct );
    } else {
        cx  = quad( (df*df + 0.25f*pow(dx+0.5f*(SE - SW), 2.f)) * mct );
        cxd = quad( (db*db + 0.25f*pow(dx+0.5f*(NE - NW), 2.f)) * mct );
    }
    delta += (cx*df - cxd*db);

    return delta;
}

float computeCurvatureBasedUpdate(const float mct,
                                  const float NW, const float N, const float NE,
                                  const float  W, const float C, const float  E,
                                  const float SW, const float S, const float SE, const af_flux_function fftype)
{
    float delta = 0.f;
    float prop_grad = 0.f;

    float df0, db0;
    float dx, dy, df, db, cx, cxd, gmf, gmb, gmsqf, gmsqb;

    // centralized derivatives
    dx = (E-W)*0.5f;
    dy = (S-N)*0.5f;

    // half-d's and conductance along first dimension
    df  = E - C;
    db  = C - W;
    df0 = df;
    db0 = db;

    if (fftype==AF_FLUX_EXPONENTIAL) {
        gmsqf = (df*df + 0.25f*pow(dy+0.5f*(SE - NE), 2.f));
        gmsqb = (db*db + 0.25f*pow(dy+0.5f*(SW - NW), 2.f));
    } else {
        gmsqf = (df*df + 0.25f*pow(dy+0.5f*(SE - NE), 2.f));
        gmsqb = (db*db + 0.25f*pow(dy+0.5f*(SW - NW), 2.f));
    }

    gmf = sqrt(1.0e-10f + gmsqf);
    gmb = sqrt(1.0e-10f + gmsqb);

    cx  = exp( gmsqf * mct );
    cxd = exp( gmsqb * mct );

    delta += ((df/gmf)*cx - (db/gmb)*cxd);

    // half-d's and conductance along second dimension
    df  = S - C;
    db  = C - N;

    if (fftype==AF_FLUX_EXPONENTIAL) {
        gmsqf = (df*df + 0.25f*pow(dx+0.5f*(SE - SW), 2.f));
        gmsqb = (db*db + 0.25f*pow(dx+0.5f*(NE - NW), 2.f));
    } else {
        gmsqf = (df*df + 0.25f*pow(dx+0.5f*(SE - SW), 2.f));
        gmsqb = (db*db + 0.25f*pow(dx+0.5f*(NE - NW), 2.f));
    }
    gmf = sqrt(1.0e-10f + gmsqf);
    gmb = sqrt(1.0e-10f + gmsqb);

    cx  = exp( gmsqf * mct );
    cxd = exp( gmsqb * mct );

    delta += ((df/gmf)*cx - (db/gmb)*cxd);

    if (delta>0){
        prop_grad += (pow(fminf(db0, 0.0f),2.0f) + pow(fmaxf(df0, 0.0f), 2.0f));
        prop_grad += (pow(fminf( db, 0.0f),2.0f) + pow(fmaxf( df, 0.0f), 2.0f));
    } else {
        prop_grad += (pow(fmaxf(db0, 0.0f),2.0f) + pow(fminf(df0, 0.0f), 2.0f));
        prop_grad += (pow(fmaxf( db, 0.0f),2.0f) + pow(fminf( df, 0.0f), 2.0f));
    }

    return sqrt(prop_grad)*delta;
}

template<typename T, bool isMCDE>
void anisotropicDiffusion(Param<T> inout, const float dt, const float mct, const af_flux_function fftype)
{
    auto dims = inout.dims();
    auto strides = inout.strides();

    for(int b3=0; b3<dims[3]; ++b3) {
        for(int b2=0; b2<dims[2]; ++b2) {

            T* img = inout.get() + b2*strides[2] + b3*strides[3];

            for(int j=1; j<dims[1]-1; ++j)
            {
                for(int i=1; i<dims[0]-1; ++i)
                {
                    float C = 0.f;
                    float delta = 0.f;

                    // int ip1 = clamp((int)i + 1, 0, (int)dims[0]-1);
                    // int im1 = clamp((int)i - 1, 0, (int)dims[0]-1);
                    // int jp1 = clamp((int)j + 1, 0, (int)dims[1]-1);
                    // int jm1 = clamp((int)j - 1, 0, (int)dims[1]-1);
                    // 400ms
                    int ip1 = i + 1;
                    int im1 = i - 1;
                    int jp1 = j + 1;
                    int jm1 = j - 1;

                    if (isMCDE) {
                        delta = computeCurvatureBasedUpdate(
                                mct,
                                img[ index(im1, jm1, strides[1]) ],
                                img[ index(i  , jm1, strides[1]) ],
                                img[ index(ip1, jm1, strides[1]) ],
                                img[ index(im1, j  , strides[1]) ],
                                C = img[ index(i  , j,  strides[1]) ],
                                img[ index(ip1, j  , strides[1]) ],
                                img[ index(im1, jp1, strides[1]) ],
                                img[ index(i  , jp1, strides[1]) ],
                                img[ index(ip1, jp1, strides[1]) ], fftype);

                    } else {
                        delta = computeGradientBasedUpdate(
                                mct,
                                img[ index(im1, jm1,  strides[1]) ],
                                img[ index(i  , jm1,  strides[1]) ],
                                img[ index(ip1, jm1,  strides[1]) ],
                                img[ index(im1, j  ,  strides[1]) ],
                                C = img[ index(i  , j,  strides[1]) ],
                                img[ index(ip1, j  ,  strides[1]) ],
                                img[ index(im1, jp1,  strides[1]) ],
                                img[ index(i  , jp1,  strides[1]) ],
                                img[ index(ip1, jp1,  strides[1]) ], fftype);
                    }

                    img[i + j*strides[1]] = (T)(C + delta*dt);
                }
            }
        }
    }
}
}
}
