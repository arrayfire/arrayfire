/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <reduce.hpp>
#include <arith.hpp>
#include <math.hpp>
#include <unary.hpp>
#include <range.hpp>
#include <reduce.hpp>
#include <transpose.hpp>

using namespace detail;

template<typename T>
Array<T> gaussianKernel(const int rows, const int cols, const double sigma_r, const double sigma_c)
{
    const dim4 odims = dim4(rows, cols);
    double sigma = 0;

    Array<T> tmp  = createValueArray<T>(odims, scalar<T>(0));
    Array<T> half = createValueArray<T>(odims, 0.5);
    Array<T> zero = createValueArray<T>(odims, scalar<T>(0));

    if (cols > 1) {
        Array<T> wt = range<T>(dim4(cols, rows), 0);
        Array<T> w  = transpose<T>(wt, false);

        Array<T> c = createValueArray<T>(odims, scalar<T>((double)(cols - 1) / 2.0));
        w = arithOp<T, af_sub_t>(w, c, odims);

        sigma = sigma_c > 0 ? sigma_c : 0.25 * cols;
        Array<T> sig = createValueArray<T>(odims, sigma);
        w = arithOp<T, af_div_t>(w, sig, odims);

        w = arithOp<T, af_mul_t>(w, w, odims);
        tmp = arithOp<T, af_add_t>(w, tmp, odims);
    }

    if (rows > 1) {
        Array<T> w = range<T>(dim4(rows, cols), 0);

        Array<T> r = createValueArray<T>(odims, scalar<T>((double)(rows - 1) / 2.0));
        w = arithOp<T, af_sub_t>(w, r, odims);

        sigma = sigma_r > 0 ? sigma_r : 0.25 * rows;
        Array<T> sig = createValueArray<T>(odims, sigma);

        w = arithOp<T, af_div_t>(w, sig, odims);
        w = arithOp<T, af_mul_t>(w, w, odims);
        tmp = arithOp<T, af_add_t>(w, tmp, odims);
    }

    tmp = arithOp<T, af_mul_t>(half, tmp, odims);
    tmp = arithOp<T, af_sub_t>(zero, tmp, odims);
    tmp = unaryOp<T, af_exp_t>(tmp);

    // Use this instead of (2 * pi * sig^2);
    // This ensures the window adds up to 1
    T norm_factor = reduce_all<af_add_t, T, T>(tmp);

    Array<T> norm = createValueArray(odims, norm_factor);
    Array<T> res = arithOp<T, af_div_t>(tmp, norm, odims);

    return res;
}

af_err af_gaussian_kernel(af_array *out,
                          const int rows, const int cols,
                          const double sigma_r, const double sigma_c)
{
    try {
        af_array res;
        res = getHandle<float>(gaussianKernel<float>(rows, cols, sigma_r, sigma_c));
        std::swap(*out, res);
    }CATCHALL;
    return AF_SUCCESS;
}
