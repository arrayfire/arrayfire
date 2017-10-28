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
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::abs;
using af::cfloat;
using af::cdouble;
using af::dtype_traits;

template<typename T>
void choleskyTester(const int n, double eps, bool is_upper)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

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

    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(real(in - re))), eps);
    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(imag(in - re))), eps);

    //! [ex_chol_inplace]
    af::array in2 = in.copy();
    choleskyInPlace(in2, is_upper);
    //! [ex_chol_inplace]

    af::array out2 = is_upper ? upper(in2) : lower(in2);

    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(real(out2 - out))), eps);
    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(imag(out2 - out))), eps);
}

template<typename T>
class Cholesky : public ::testing::Test
{

};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_CASE(Cholesky, TestTypes);

template<typename T>
double eps();

template<>
double eps<float>() {
    return 0.05f;
}

template<>
double eps<double>() {
    return 1e-8;
}

template<>
double eps<cfloat>() {
    return 0.05f;
}

template<>
double eps<cdouble>() {
    return 1e-8;
}

TYPED_TEST(Cholesky, Upper) {
    choleskyTester<TypeParam>( 500, eps<TypeParam>(), true );
}

TYPED_TEST(Cholesky, UpperLarge) {
    choleskyTester<TypeParam>( 1000, eps<TypeParam>(), true );
}

TYPED_TEST(Cholesky, UpperMultipleOfTwo) {
    choleskyTester<TypeParam>( 512, eps<TypeParam>(), true );
}

TYPED_TEST(Cholesky, UpperMultipleOfTwoLarge) {
    choleskyTester<TypeParam>( 1024, eps<TypeParam>(), true );
}

TYPED_TEST(Cholesky, Lower) {
    choleskyTester<TypeParam>( 500, eps<TypeParam>(), false );
}

TYPED_TEST(Cholesky, LowerLarge) {
    choleskyTester<TypeParam>( 1000, eps<TypeParam>(), false );
}

TYPED_TEST(Cholesky, LowerMultipleOfTwo) {
    choleskyTester<TypeParam>( 512, eps<TypeParam>(), false );
}

TYPED_TEST(Cholesky, LowerMultipleOfTwoLarge) {
    choleskyTester<TypeParam>( 1024, eps<TypeParam>(), false );
}
