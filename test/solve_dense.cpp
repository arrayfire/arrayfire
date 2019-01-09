/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// NOTE: Tests are known to fail on OSX when utilizing the CPU and OpenCL
// backends for sizes larger than 128x128 or more. You can read more about it on
// issue https://github.com/arrayfire/arrayfire/issues/1617

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <thread>
#include "solve_common.hpp"

template <typename T>
class Solve : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_CASE(Solve, TestTypes);

template <typename T>
double eps();

template <>
double eps<float>() {
    return 0.01f;
}

template <>
double eps<double>() {
    return 1e-5;
}

template <>
double eps<cfloat>() {
    return 0.01f;
}

template <>
double eps<cdouble>() {
    return 1e-5;
}

TYPED_TEST(Solve, Square) {
    solveTester<TypeParam>(100, 100, 10, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareMultipleOfTwo) {
    solveTester<TypeParam>(96, 96, 16, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareLarge) {
    solveTester<TypeParam>(1000, 1000, 10, eps<TypeParam>());
}

TYPED_TEST(Solve, SquareMultipleOfTwoLarge) {
    solveTester<TypeParam>(2048, 2048, 32, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDetermined) {
    solveTester<TypeParam>(80, 100, 20, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDeterminedMultipleOfTwo) {
    solveTester<TypeParam>(96, 128, 40, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDeterminedLarge) {
    solveTester<TypeParam>(800, 1000, 200, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresUnderDeterminedMultipleOfTwoLarge) {
    solveTester<TypeParam>(1536, 2048, 400, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDetermined) {
    solveTester<TypeParam>(80, 60, 20, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDeterminedMultipleOfTwo) {
    solveTester<TypeParam>(96, 64, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDeterminedLarge) {
    solveTester<TypeParam>(800, 600, 64, eps<TypeParam>());
}

TYPED_TEST(Solve, LeastSquaresOverDeterminedMultipleOfTwoLarge) {
    solveTester<TypeParam>(1536, 1024, 1, eps<TypeParam>());
}

TYPED_TEST(Solve, LU) { solveLUTester<TypeParam>(100, 10, eps<TypeParam>()); }

TYPED_TEST(Solve, LUMultipleOfTwo) {
    solveLUTester<TypeParam>(96, 64, eps<TypeParam>());
}

TYPED_TEST(Solve, LULarge) {
    solveLUTester<TypeParam>(1000, 100, eps<TypeParam>());
}

TYPED_TEST(Solve, LUMultipleOfTwoLarge) {
    solveLUTester<TypeParam>(2048, 512, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpper) {
    solveTriangleTester<TypeParam>(100, 10, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpperMultipleOfTwo) {
    solveTriangleTester<TypeParam>(96, 64, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpperLarge) {
    solveTriangleTester<TypeParam>(1000, 100, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleUpperMultipleOfTwoLarge) {
    solveTriangleTester<TypeParam>(2048, 512, true, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLower) {
    solveTriangleTester<TypeParam>(100, 10, false, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLowerMultipleOfTwo) {
    solveTriangleTester<TypeParam>(96, 64, false, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLowerLarge) {
    solveTriangleTester<TypeParam>(1000, 100, false, eps<TypeParam>());
}

TYPED_TEST(Solve, TriangleLowerMultipleOfTwoLarge) {
    solveTriangleTester<TypeParam>(2048, 512, false, eps<TypeParam>());
}

#if !defined(AF_OPENCL)
int nextTargetDeviceId() {
    static int nextId = 0;
    return nextId++;
}

#define SOLVE_LU_TESTS_THREADING(T, eps)                              \
    tests.emplace_back(solveLUTester<T>, 1000, 100, eps,              \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTriangleTester<T>, 1000, 100, true, eps,  \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTriangleTester<T>, 1000, 100, false, eps, \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTester<T>, 1000, 1000, 100, eps,          \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTester<T>, 800, 1000, 200, eps,           \
                       nextTargetDeviceId() % numDevices);            \
    tests.emplace_back(solveTester<T>, 800, 600, 64, eps,             \
                       nextTargetDeviceId() % numDevices);

TEST(Solve, Threading) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&numDevices));
    ASSERT_EQ(true, numDevices > 0);

    SOLVE_LU_TESTS_THREADING(float, 0.01);
    SOLVE_LU_TESTS_THREADING(cfloat, 0.01);
    if (noDoubleTests<double>()) {
        SOLVE_LU_TESTS_THREADING(double, 1E-5);
        SOLVE_LU_TESTS_THREADING(cdouble, 1E-5);
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

#undef SOLVE_LU_TESTS_THREADING
#endif
#undef SOLVE_TESTS
