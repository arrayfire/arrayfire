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

template<typename T>
void choleskyTester(const int n, double eps, bool is_upper)
{
    if (noDoubleTests<T>()) return;

    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    // Prepare positive definite matrix
#if 1
    af::array a = cpu_randu<T>(af::dim4(n, n));
#else
    af::array a = af::randu(n, n, ty);
#endif
    af::array b = 10 * n * af::identity(n, n, ty);
    af::array in = matmul(a.H(), a) + b;

    //! [ex_chol_reg]
    af::array out;
    cholesky(out, in, is_upper);
    //! [ex_chol_reg]

    af::array re = is_upper ? matmul(out.H(), out) : matmul(out, out.H());

    ASSERT_NEAR(0, af::max<double>(af::abs(real(in - re))), eps);
    ASSERT_NEAR(0, af::max<double>(af::abs(imag(in - re))), eps);

    //! [ex_chol_inplace]
    af::array in2 = in.copy();
    choleskyInPlace(in2, is_upper);
    //! [ex_chol_inplace]

    af::array out2 = is_upper ? upper(in2) : lower(in2);

    ASSERT_NEAR(0, af::max<double>(af::abs(real(out2 - out))), eps);
    ASSERT_NEAR(0, af::max<double>(af::abs(imag(out2 - out))), eps);
}

#define CHOLESKY_BIG_TESTS(T, eps)              \
    TEST(Cholesky, T##Upper)                    \
    {                                           \
        choleskyTester<T>( 500, eps, true );    \
    }                                           \
    TEST(Cholesky, T##Lower)                    \
    {                                           \
        choleskyTester<T>(1000, eps, false);    \
    }                                           \
    TEST(Cholesky, T##UpperMultiple)            \
    {                                           \
        choleskyTester<T>(1024, eps, true );    \
    }                                           \
    TEST(Cholesky, T##LowerMultiple)            \
    {                                           \
        choleskyTester<T>( 512, eps, false);    \
    }                                           \


CHOLESKY_BIG_TESTS(float, 0.05)
CHOLESKY_BIG_TESTS(double, 1E-8)
CHOLESKY_BIG_TESTS(cfloat, 0.05)
CHOLESKY_BIG_TESTS(cdouble, 1E-8)
