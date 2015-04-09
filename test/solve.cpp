/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

///////////////////////////////// CPP ////////////////////////////////////
//
const double max_dbl_err = 1E-10;
TEST(Solve, Square_CPP)
{
    if (noDoubleTests<double>()) return;

    const int M = 1000;
    const int N = M;
    const int K = 100;
    af::array a = af::randu(M, N, f64);
    af::array x0 = af::randu(N, K, f64);
    af::array b0 = af::matmul(a, x0);

    af::array x1 = af::solve(a, b0);
    af::array b1 = af::matmul(a, x1);

    ASSERT_EQ(af::sum<double>(abs(b0 - b1)) / (M * K) < max_dbl_err, true);
}

TEST(Solve, UnderDetermined_CPP)
{
    if (noDoubleTests<double>()) return;

    const int M = 800;
    const int N = 1000;
    const int K = 100;
    af::array a = af::randu(M, N, f64);
    af::array x0 = af::randu(N, K, f64);
    af::array b0 = af::matmul(a, x0);

    af::array x1 = af::solve(a, b0);
    af::array b1 = af::matmul(a, x1);

    ASSERT_EQ(af::sum<double>(abs(b0 - b1)) / (M * K) < max_dbl_err, true);
}

TEST(Solve, OverDetermined_CPP)
{
    if (noDoubleTests<double>()) return;

    const int M = 1000;
    const int N = 800;
    const int K = 100;
    af::array a = af::randu(M, N, f64);
    af::array x0 = af::randu(N, K, f64);
    af::array b0 = af::matmul(a, x0);

    af::array x1 = af::solve(a, b0);
    af::array b1 = af::matmul(a, x1);

    ASSERT_EQ(af::sum<double>(abs(b0 - b1)) / (M * K) < max_dbl_err, true);
}
