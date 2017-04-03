/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include "solve_common.hpp"

#define SOLVE_LU_TESTS(T, eps)                          \
    TEST(SOLVE_LU, T##Reg)                              \
    {                                                   \
        solveLUTester<T>(1000, 100, eps);               \
    }                                                   \
    TEST(SOLVE_LU, T##RegMultiple)                      \
    {                                                   \
        solveLUTester<T>(2048, 512, eps);               \
    }                                                   \


#define SOLVE_TRIANGLE_TESTS(T, eps)                    \
    TEST(SOLVE_Upper, T##Reg)                           \
    {                                                   \
        solveTriangleTester<T>(1000, 100, true, eps);   \
    }                                                   \
    TEST(SOLVE_Upper, T##RegMultiple)                   \
    {                                                   \
        solveTriangleTester<T>(2048, 512, true, eps);   \
    }                                                   \
    TEST(SOLVE_Lower, T##Reg)                           \
    {                                                   \
        solveTriangleTester<T>(1000, 100, false, eps);  \
    }                                                   \
    TEST(SOLVE_Lower, T##RegMultiple)                   \
    {                                                   \
        solveTriangleTester<T>(2048, 512, false, eps);  \
    }                                                   \

#define SOLVE_GENERAL_TESTS(T, eps)                     \
    TEST(SOLVE, T##Square)                              \
    {                                                   \
        solveTester<T>(1000, 1000, 100, eps);           \
    }                                                   \
    TEST(SOLVE, T##SquareMultiple)                      \
    {                                                   \
        solveTester<T>(2048, 2048, 512, eps);           \
    }                                                   \

#define SOLVE_LEASTSQ_TESTS(T, eps)                     \
    TEST(SOLVE, T##RectUnder)                           \
    {                                                   \
        solveTester<T>(800, 1000, 200, eps);            \
    }                                                   \
    TEST(SOLVE, T##RectUnderMultiple)                   \
    {                                                   \
        solveTester<T>(1536, 2048, 400, eps);           \
    }                                                   \
    TEST(SOLVE, T##RectOver)                            \
    {                                                   \
        solveTester<T>(800, 600, 64, eps);              \
    }                                                   \
    TEST(SOLVE, T##RectOverMultiple)                    \
    {                                                   \
        solveTester<T>(1536, 1024, 1, eps);             \
    }                                                   \

#define SOLVE_TESTS(T, eps)                             \
    SOLVE_GENERAL_TESTS(T, eps)                         \
    SOLVE_LEASTSQ_TESTS(T, eps)                         \
    SOLVE_LU_TESTS(T, eps)                              \
    SOLVE_TRIANGLE_TESTS(T, eps)                        \


SOLVE_TESTS(float, 0.01)
SOLVE_TESTS(double, 1E-5)
SOLVE_TESTS(cfloat, 0.01)
SOLVE_TESTS(cdouble, 1E-5)

#undef SOLVE_TESTS
