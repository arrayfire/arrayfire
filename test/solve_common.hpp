/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <arrayfire.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::cdouble;
using af::cfloat;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
void solveTester(const int m, const int n, const int k, double eps,
                 int targetDevice = -1) {
    if (targetDevice >= 0) af::setDevice(targetDevice);

    af::deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

#if 1
    af::array A  = cpu_randu<T>(af::dim4(m, n));
    af::array X0 = cpu_randu<T>(af::dim4(n, k));
#else
    af::array A  = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array X0 = af::randu(n, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif
    af::array B0 = af::matmul(A, X0);

    //! [ex_solve]
    af::array X1 = af::solve(A, B0);
    //! [ex_solve]

    //! [ex_solve_recon]
    af::array B1 = af::matmul(A, X1);
    //! [ex_solve_recon]

    ASSERT_ARRAYS_NEAR(B0, B1, eps);
}

template<typename T>
void solveLUTester(const int n, const int k, double eps,
                   int targetDevice = -1) {
    if (targetDevice >= 0) af::setDevice(targetDevice);

    af::deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

#if 1
    af::array A  = cpu_randu<T>(af::dim4(n, n));
    af::array X0 = cpu_randu<T>(af::dim4(n, k));
#else
    af::array A  = af::randu(n, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array X0 = af::randu(n, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif
    af::array B0 = af::matmul(A, X0);

    //! [ex_solve_lu]
    af::array A_lu, pivot;
    af::lu(A_lu, pivot, A);
    af::array X1 = af::solveLU(A_lu, pivot, B0);
    //! [ex_solve_lu]

    af::array B1 = af::matmul(A, X1);

    ASSERT_ARRAYS_NEAR(B0, B1, eps);
}

template<typename T>
void solveTriangleTester(const int n, const int k, bool is_upper, double eps,
                         int targetDevice = -1) {
    if (targetDevice >= 0) af::setDevice(targetDevice);

    af::deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

#if 1
    af::array A  = cpu_randu<T>(af::dim4(n, n));
    af::array X0 = cpu_randu<T>(af::dim4(n, k));
#else
    af::array A  = af::randu(n, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array X0 = af::randu(n, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    af::array L, U, pivot;
    af::lu(L, U, pivot, A);

    af::array AT = is_upper ? U : L;
    af::array B0 = af::matmul(AT, X0);
    af::array X1;

    if (is_upper) {
        //! [ex_solve_upper]
        af::array X = af::solve(AT, B0, AF_MAT_UPPER);
        //! [ex_solve_upper]

        X1 = X;
    } else {
        //! [ex_solve_lower]
        af::array X = af::solve(AT, B0, AF_MAT_LOWER);
        //! [ex_solve_lower]

        X1 = X;
    }

    af::array B1 = af::matmul(AT, X1);

    ASSERT_ARRAYS_NEAR(B0, B1, eps);
}
