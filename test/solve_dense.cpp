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

template<typename T>
void solveTester(const int m, const int n, const int k, double eps)
{
    if (noDoubleTests<T>()) return;
#if 1
    af::array A  = cpu_randu<T>(af::dim4(m, n));
    af::array X0 = cpu_randu<T>(af::dim4(n, k));
#else
    af::array A  = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array X0 = af::randu(n, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif
    af::array B0 = af::matmul(A, X0);

    af::array X1 = af::solve(A, B0);
    af::array B1 = af::matmul(A, X1);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(B0 - B1))) / (m * k), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(B0 - B1))) / (m * k), eps);
}

#define SOLVE_TESTS(T, eps)                     \
    TEST(SOLVE, T##Square)                      \
    {                                           \
        solveTester<T>(1000, 1000, 100, eps);   \
    }                                           \
    TEST(SOLVE, T##SquareMultiple)              \
    {                                           \
        solveTester<T>(2048, 2048, 512, eps);   \
    }                                           \
    TEST(SOLVE, T##RectUnder)                   \
    {                                           \
        solveTester<T>(800, 1000, 200, eps);    \
    }                                           \
    TEST(SOLVE, T##RectUnderMultiple)           \
    {                                           \
        solveTester<T>(1536, 2048, 400, eps);   \
    }                                           \
    TEST(SOLVE, T##RectOver)                    \
    {                                           \
        solveTester<T>(800, 600, 50, eps);      \
    }                                           \
    TEST(SOLVE, T##RectOverMultiple)            \
    {                                           \
        solveTester<T>(1536, 1024, 1, eps);     \
    }

SOLVE_TESTS(float, 0.01)
SOLVE_TESTS(double, 1E-5)
SOLVE_TESTS(cfloat, 0.01)
SOLVE_TESTS(cdouble, 1E-5)
